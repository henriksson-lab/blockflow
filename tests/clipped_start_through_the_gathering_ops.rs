// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **`StepOrigin::ClippedStart` through the per-voxel neighbourhood ops**, which
// is the half of that parameter `tests/stepped_element_clipped_start.rs` does
// not cover: that file takes the element through `ops::local`'s sampled
// statistic, and this one takes it through the ops that gather a window at every
// voxel.
//
// What the two origins are is stated at `StepOrigin` and derived at
// `StructuringElement::offsets_at`; in one line, a decimation counted from the
// clipped start of the window re-phases where the window meets a **low face of
// the volume**, so the element reads a different set of offsets there and
// `offsets()` is the interior case only.
//
// Every op is one of three things, and this file says which
// ---------------------------------------------------------
// An op handed such an element may **honour** it — ask `offsets_at` per voxel —
// or **refuse** it by name, or be **unaffected**, which is what every op that
// only ever sees an unstepped element is: a stride of one over the clipped range
// visits every coordinate of it, so `StructuringElement` normalises that element
// to `StepOrigin::Anchor` and the two rules are one rule. What is not available
// is computing the anchored window under the other one's name, which is what
// reading `offsets()` does and is what this file exists to keep closed.
//
// | op | what it does | where it is asserted |
// |---|---|---|
// | `ops::rank`, plain and masked | honours | here, and in its own tests |
// | `ops::morphology`, all four | honours | here, and `the_extreme_ranks_agree_over_a_re_phasing_element_too` |
// | `ops::reconstruct`'s step | honours | its own tests |
// | `ops::background` | honours, by being two rank filters and a subtraction | its own tests |
// | `ops::lattice`'s windowed statistic | honours, at each sample's position in the volume | at the end of this file |
// | `ops::label`'s stamp | honours, at the point's position in the volume | at the end of this file, and its own tests |
// | `ops::voxelize`'s deposit | honours, at the point's position in the volume | at the end of this file, and its own tests |
// | `ops::sliding` | **refuses**, by name | its own tests |
//
// The refusal is the interesting one and it is not a shortfall. A histogram
// carried along a scan line is a decomposition of **one** window into what a
// step of one retires and admits; a window that re-phases has no such
// decomposition, because consecutive centres read different residue classes.
// Gathering the interior offsets there would be the silent wrong window, and the
// op is documented as byte-identical to the dense filter — so it refuses, and
// that claim holds over every element it accepts.
//
// The two that **stamp** rather than gather
// -----------------------------------------
// `ops::label` and `ops::voxelize` place an element around a *point* and write,
// where every op above reads a neighbourhood around a voxel. That is the same
// question only if a stamp is a gather's transpose, and it is: the transpose of
// `out[c] = f({in[c + o] : o in K(c)})` scatters `p + o` for `o` in `K(p)`, the
// kernel read at the **source**, which for those two ops is the point. Underneath
// it, `ClippedStart` is a property of a *(position, volume)* pair rather than of
// a direction of data flow, and both ops write into the volume their points live
// in, so the clip is one clip. The last test of this file is where that is
// measured; each op's module header carries the argument.
//
// What the sweep in the middle establishes
// -----------------------------------------
// **Decomposition invariance**, byte for byte, over every block size and split,
// for an op whose window depends on where it is evaluated. The argument is that
// a halo of at least the element's low side makes a window clipped at a buffer's
// low edge a window clipped at the volume's — unless the buffer starts at zero,
// where its low edge *is* the volume's face. The argument holds only while the
// halo does, so it is measured rather than relied on.

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, Chain};
use blockflow::ops::voxelize::{voxelize_into, Point};
use blockflow::ops::{
    label_points_into, lattice_statistic_into, ElementShape, Morphology, MorphologyOp, Rank,
    RankFilterOp, SampleLattice, Sampling, StepOrigin, StructuringElement, MAX_EXACT_LABEL,
};
use blockflow::region::Region;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Long enough on axis 0 for several blocks and **narrower there than the
/// window**, so that every anchor on that axis re-phases and no block holds only
/// interior voxels.
const VOLUME: [usize; 3] = [24, 6, 4];
const SIZE: [usize; 3] = [11, 3, 1];
const STEP: [usize; 3] = [2, 2, 1];

fn clipped() -> StructuringElement {
    StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        SIZE,
        STEP,
        StepOrigin::ClippedStart,
    )
    .unwrap()
}

fn anchored() -> StructuringElement {
    StructuringElement::from_size_stepped_at(ElementShape::Box, SIZE, STEP, StepOrigin::Anchor)
        .unwrap()
}

fn image() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        // deterministic, not smooth, and different on each axis, so a window that
        // moved by one voxel cannot give the same answer by accident
        let value = (i * 37 + j * 11 + k * 5) % 29;
        value as f64 + (i % 2) as f64 * 0.5
    })
}

/// A mask volume, for the ops that take one.
fn mask() -> Array3<bool> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        (i * 5 + j * 3 + k) % 4 != 0
    })
}

/// The ops under test: the dense filter at both truncation conventions, and the
/// two morphological forms — a primitive and the composition that reads twice as
/// far.
///
/// A tag rather than a list of built chains, because a `Chain` is not `Clone` and
/// each block layout below needs one of its own; building it from the tag is what
/// keeps every layout running the same op.
#[derive(Debug, Clone, Copy)]
enum Case {
    Median,
    Percentile,
    Erode,
    Open,
}

impl Case {
    const ALL: [Case; 4] = [Case::Median, Case::Percentile, Case::Erode, Case::Open];

    /// `Bool` for the morphology, which is what that op is for; `f64` for the
    /// filter, where a differing bit is a differing value rather than a flipped
    /// flag.
    fn dtype(self) -> Dtype {
        match self {
            Case::Median | Case::Percentile => Dtype::F64,
            Case::Erode | Case::Open => Dtype::Bool,
        }
    }

    fn chain(self, element: &StructuringElement) -> Chain {
        let element = element.clone();
        match self {
            Case::Median => Chain::op(RankFilterOp::median("median", element)),
            Case::Percentile => Chain::op(RankFilterOp::new(
                "percentile",
                element,
                Rank::ceiling_percentile(0.25).unwrap(),
            )),
            Case::Erode => Chain::op(MorphologyOp::new("erode", Morphology::Erode, element)),
            Case::Open => Chain::op(MorphologyOp::new("open", Morphology::Open, element)),
        }
    }
}

fn source(dtype: Dtype) -> Voxels {
    match dtype {
        Dtype::Bool => mask().into(),
        _ => image().into(),
    }
}

fn bits(out: &Voxels) -> Vec<u64> {
    match out.dtype() {
        Dtype::Bool => out
            .view::<bool>()
            .unwrap()
            .iter()
            .map(|&flag| flag as u64)
            .collect(),
        _ => out
            .view::<f64>()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    }
}

fn whole_volume(case: Case, element: &StructuringElement) -> Voxels {
    let dtype = case.dtype();
    let mut out = Voxels::zeros(dtype, VOLUME).unwrap();
    case.chain(element)
        .apply(&source(dtype), &mut out, &Anchor::whole(VOLUME))
        .expect("the whole-volume reference must run");
    out
}

fn blocked(case: Case, element: &StructuringElement, grid: &BlockGrid) -> Voxels {
    let dtype = case.dtype();
    let chain = case.chain(element);
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
        dtype,
        phases: vec![phase],
        chain_reach: chain.reach3(&VOLUME),
    };
    decomposition.check().expect("an honest plan must tile");
    let workflow = Workflow::new(chain, VOLUME, dtype);
    let env = ArrayEnvironment::new(source(dtype), decomposition.n_phases(), [4, 4, 4])
        .expect("an environment");
    execute(
        "clipped-start",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().clone()
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

// ---------------------------------------- the ops that honour the origin --

/// **Byte-identical to the whole-volume answer under every decomposition**, for
/// every op that gathers a window at every voxel, on an element whose membership
/// depends on where it is evaluated.
///
/// This is the property the crate exists for, and it is the one an origin keyed
/// on the wrong extent would break first: a rule that re-phased at a *buffer's*
/// low edge would compute a different filter in every block layout, and every
/// block layout below has different edges.
#[test]
fn every_decomposition_gives_the_whole_volume_answer() {
    let element = clipped();
    let mut checked = 0;
    for case in Case::ALL {
        let want = bits(&whole_volume(case, &element));
        for grid in grids() {
            let got = bits(&blocked(case, &element, &grid));
            let differing = got
                .iter()
                .zip(want.iter())
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(
                differing,
                0,
                "{case:?}: block {:?} disagreed with the whole-volume answer at {differing} voxels",
                grid.block()
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 4 * 8);
}

/// The sweep above is not an invariance test of the anchored element wearing
/// another name: the two origins compute **different volumes** here, through
/// every one of these ops.
#[test]
fn the_two_origins_are_different_operations_through_every_op_that_honours_them() {
    let clipped = clipped();
    let anchored = anchored();
    assert_ne!(clipped, anchored);
    for case in Case::ALL {
        let left = bits(&whole_volume(case, &clipped));
        let right = bits(&whole_volume(case, &anchored));
        let differing = left
            .iter()
            .zip(right.iter())
            .filter(|(left, right)| left != right)
            .count();
        assert!(
            differing > 0,
            "{case:?}: the two origins gave the same volume, so the invariance sweep is a sweep \
             over an element that quietly behaves like the anchored one"
        );
    }
}

/// **The anchored element did not move**, at the level a caller sees: an element
/// whose offsets are one set gives the same answer whether it is asked as the
/// whole volume or as a block of one, through every op here, and that is what
/// every existing caller of this crate depends on.
///
/// The unit tests in each module assert the same thing against the definition;
/// this asserts it through the executor, which is the path a chain takes.
#[test]
fn the_anchored_answer_is_unchanged_under_decomposition_too() {
    let element = anchored();
    assert_eq!(element.origin(), StepOrigin::Anchor);
    for case in Case::ALL {
        let want = bits(&whole_volume(case, &element));
        for grid in grids() {
            assert_eq!(
                bits(&blocked(case, &element, &grid)),
                want,
                "{case:?}: block {:?}",
                grid.block()
            );
        }
    }
}

/// An **unstepped** element normalises to [`StepOrigin::Anchor`], whichever
/// origin was asked for, so an op that only ever sees one is unaffected by this
/// whole question. Asserted rather than argued, because it is what makes the
/// third of the three answers — "genuinely irrelevant" — a fact about the type
/// rather than a hope about callers.
#[test]
fn an_unstepped_element_is_the_same_element_under_either_origin() {
    for shape in [
        ElementShape::Box,
        ElementShape::Ellipsoid,
        ElementShape::ExtentEllipsoid,
    ] {
        for size in [[7, 5, 3], [8, 8, 1], [1, 1, 1]] {
            let asked = StructuringElement::from_size_stepped_at(
                shape,
                size,
                [1, 1, 1],
                StepOrigin::ClippedStart,
            )
            .unwrap();
            let plain = StructuringElement::from_size(shape, size).unwrap();
            assert_eq!(asked.origin(), StepOrigin::Anchor, "{shape:?} {size:?}");
            assert_eq!(asked, plain, "{shape:?} {size:?}");
        }
    }
}

// --------------------------------- the ops that were closed last, measured --

/// **`ops::lattice`'s windowed statistic gathers the window its element names.**
///
/// This assertion used to read the other way. It pinned the gap — the op read
/// `StructuringElement::offsets`, one set at every sample, so a lattice
/// statistic over a re-phasing element computed the *anchored* filter — and it
/// said in as many words that closing the gap would fail it and that it should
/// then be **inverted rather than deleted**. The close was
/// `offsets_at(centre, lattice.volume(), &mut scratch)` in the op's own loop,
/// which already had the sample centre in volume coordinates and already held
/// the volume on the lattice. So this is that inversion.
///
/// It now states the rule twice over: the statistic **is** the `offsets_at`
/// gather, written out here rather than called through the op, and it **is not**
/// the anchored gather on this lattice. The second half is what keeps the first
/// from passing vacuously — an element whose two origins happen to agree would
/// satisfy any implementation.
#[test]
fn the_lattice_statistic_gathers_the_window_the_origin_names() {
    let volume = [24usize, 1, 1];
    let input = Array3::from_shape_fn((volume[0], 1, 1), |(i, _, _)| ((i * 37) % 29) as f64);
    let lattice = SampleLattice::of(&Sampling::every([5, 1, 1]), volume).unwrap();
    let counts = [lattice.count(0), lattice.count(1), lattice.count(2)];

    let sampled = |element: &StructuringElement| -> Array3<f64> {
        let mut out = Array3::<f64>::zeros((counts[0], counts[1], counts[2]));
        lattice_statistic_into(
            input.view(),
            [0, 0, 0],
            element,
            &lattice,
            [0, 0, 0],
            |values: &mut [f64]| values.iter().sum::<f64>() / values.len() as f64,
            out.view_mut(),
        )
        .unwrap();
        out
    };

    let element = StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        [11, 1, 1],
        [2, 1, 1],
        StepOrigin::ClippedStart,
    )
    .unwrap();
    let plain = StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        [11, 1, 1],
        [2, 1, 1],
        StepOrigin::Anchor,
    )
    .unwrap();

    // The elements are not the same element, and the first sample centre is a
    // place where they genuinely read different sets — without which the
    // comparison below would be vacuous rather than a pin.
    assert_ne!(element, plain);
    let first = lattice.centre(0, 0) as isize;
    let mut scratch = Vec::new();
    assert_ne!(
        element
            .offsets_at([first, 0, 0], volume, &mut scratch)
            .to_vec(),
        plain
            .offsets()
            .iter()
            .copied()
            .filter(|offset| {
                let at = first + offset[0];
                at >= 0 && (at as usize) < volume[0]
            })
            .collect::<Vec<_>>(),
        "the sample at {first} must be one the two origins disagree at"
    );

    // The rule, written out rather than called through the op: at each sample,
    // the window is `offsets_at` at that sample's position **in the volume**,
    // clipped to what the array holds.
    let mut expected = Array3::<f64>::zeros((counts[0], counts[1], counts[2]));
    for p in 0..counts[0] {
        let centre = lattice.centre(0, p) as isize;
        let mut window = Vec::new();
        for step in element.offsets_at([centre, 0, 0], volume, &mut scratch) {
            let at = centre + step[0];
            if at >= 0 && (at as usize) < volume[0] {
                window.push(input[[at as usize, 0, 0]]);
            }
        }
        expected[[p, 0, 0]] = window.iter().sum::<f64>() / window.len() as f64;
    }
    assert_eq!(
        sampled(&element),
        expected,
        "the lattice statistic must gather the window `offsets_at` names at each sample"
    );

    // And that is a different filter from the anchored one on this lattice —
    // without which the assertion above would hold for either implementation.
    assert_ne!(
        sampled(&element),
        sampled(&plain),
        "the two origins must disagree here, or this fixture cannot tell them apart"
    );
}

/// **`ops::voxelize` and `ops::label` place the window the origin names**, at the
/// point's own position in the volume.
///
/// This assertion used to read the other way. It pinned the gap — both ops read
/// `StructuringElement::offsets`, one set wherever the element was placed, so a
/// re-phasing element stamped the *interior* window at a point near a low face —
/// and it said in as many words that closing the gap would fail it and that it
/// should then be **inverted rather than deleted**. So this is that inversion.
///
/// **Why a stamp asks a gather's question.** Neither op gathers: they place the
/// element around a *point* and write, where a filter reads a neighbourhood
/// around a voxel. Those are the same question only if the stamp is the gather's
/// transpose, and it is. A gather is `out[c] = f({in[c + o] : o in K(c)})`, whose
/// incidence has `M[c, c + o] != 0`; the transpose of that is a scatter that
/// reads the kernel at the **source** index and writes `p + o` for `o` in `K(p)`.
/// So evaluating the element at the point is `M^T` and evaluating it anywhere
/// else is a different operator. Underneath that, `ClippedStart` is a property of
/// a *(position, volume)* pair — the slice `a[max(0, c - lo) : min(c + hi + 1, n)
/// : step]`, which says nothing about reading or writing — and both ops deposit
/// into the same volume their points live in, so `n` is one number on both sides
/// and the two questions cannot come apart. Each op's module header carries the
/// argument in full.
///
/// Stated here through **runs of both ops** rather than about the element alone,
/// which is what the old form could not do, with the rule written out from the
/// definition beside them and a negative control under it.
#[test]
fn the_stamping_ops_place_the_window_the_origin_names() {
    let element = StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        [11, 1, 1],
        [2, 1, 1],
        StepOrigin::ClippedStart,
    )
    .unwrap();
    let anchored = StructuringElement::from_size_stepped_at(
        ElementShape::Box,
        [11, 1, 1],
        [2, 1, 1],
        StepOrigin::Anchor,
    )
    .unwrap();
    let volume = [24usize, 1, 1];
    let grid = BlockGrid::whole(volume).unwrap();
    let window = Region::whole(&volume);
    let mut scratch = Vec::new();

    // The two facts the old pin recorded, kept: deep inside the volume the two
    // sets are one set, and within `lo` of the low face they are not — without
    // which there would be nothing for either op to be right or wrong about.
    assert_eq!(
        element.offsets_at([12, 0, 0], volume, &mut scratch),
        element.offsets()
    );
    let (low_side, _) = element.sides(0);
    let near = (low_side as isize) - 1;
    assert_ne!(
        element
            .offsets_at([near, 0, 0], volume, &mut scratch)
            .to_vec(),
        element
            .offsets()
            .iter()
            .copied()
            .filter(|offset| {
                let at = near + offset[0];
                at >= 0 && (at as usize) < volume[0]
            })
            .collect::<Vec<_>>(),
        "if these agreed there would be nothing for the stamping ops to be wrong about"
    );

    // Which voxels each op wrote, for a single point at `centre`.
    let stamped = |element: &StructuringElement, centre: usize| -> Vec<usize> {
        let mut out = Array3::<u64>::zeros((volume[0], 1, 1));
        label_points_into(
            &[([0, 0, 0], vec![Point::weighted([centre, 0, 0], 7.0)])],
            &grid,
            element,
            &window,
            MAX_EXACT_LABEL,
            out.view_mut(),
        )
        .unwrap();
        (0..volume[0]).filter(|&i| out[[i, 0, 0]] == 7).collect()
    };
    let deposited = |element: &StructuringElement, centre: usize| -> Vec<usize> {
        let mut out = Array3::<f64>::zeros((volume[0], 1, 1));
        voxelize_into(
            &[([0, 0, 0], vec![Point::weighted([centre, 0, 0], 2.0)])],
            &grid,
            element,
            &window,
            out.view_mut(),
        )
        .unwrap();
        (0..volume[0]).filter(|&i| out[[i, 0, 0]] == 2.0).collect()
    };

    // The rule, written out in the arithmetic it is stated in rather than asked
    // of the element: `a[max(0, c - lo) : min(c + hi + 1, n) : step]`.
    let reference = |centre: usize| -> Vec<usize> {
        (centre.saturating_sub(5)..(centre + 6).min(volume[0]))
            .step_by(2)
            .collect()
    };

    for centre in 0..volume[0] {
        assert_eq!(
            stamped(&element, centre),
            reference(centre),
            "`ops::label` stamped a window the origin does not name, at {centre}"
        );
        assert_eq!(
            deposited(&element, centre),
            reference(centre),
            "`ops::voxelize` deposited into a window the origin does not name, at {centre}"
        );
    }

    // **The negative control.** The same program with one thing changed: at a
    // point inside the low face the anchored element is a different set through
    // both ops — disjoint from it, here — so the sweep above is about the origin
    // and not about an element whose two readings happen to agree.
    let at = 2usize;
    assert_eq!(stamped(&element, at), vec![0, 2, 4, 6]);
    assert_eq!(stamped(&anchored, at), vec![1, 3, 5, 7]);
    assert_eq!(deposited(&element, at), stamped(&element, at));
    assert_eq!(deposited(&anchored, at), stamped(&anchored, at));
}
