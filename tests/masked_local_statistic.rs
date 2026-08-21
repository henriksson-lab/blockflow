// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A windowed statistic whose window has a population**, at a spacing of one
// and on a coarse lattice.
//
// The dense selection has had a population for a while: `MaskedRankFilterOp`
// reads a second image over the same window it reads its input over, and
// `SlidingHistogramOp` does the same thing through a different traversal. The
// statistic evaluated on a lattice had no population concept at all, so a window
// could not drop a voxel — which is the one thing standing between this crate
// and a masked percentile taken on a coarse grid.
//
// What this file asserts, in the order the claims depend on each other:
//
// 1. **An all-true population is the unmasked statistic**, bit for bit — so the
//    masked kernel is the same kernel and not a second one that agrees
//    approximately.
// 2. **The population changes the answer**, so nothing below is vacuous. A mask
//    that masks nothing tests nothing, and that is a mistake this crate has
//    already made once.
// 3. **Byte-identity with the dense masked selection.** At a spacing of one the
//    lattice is every voxel, the interpolation is the identity, and a
//    `Statistic::Rank` is the same order statistic the dense filter selects —
//    so the two are *the same function* and the strongest available check is
//    that they agree on the bits. Asserted over five populations, both policies
//    at the centre, and an element that does not contain its own centre.
// 4. **Byte-identity with the sliding histogram** over the same function, which
//    is a third traversal of the same definition. Three implementations agreeing
//    bit for bit is what a convention is.
// 5. **The two conditions stay distinct.** "The centre is out of the population"
//    and "the window came out empty" coincide for an element holding its own
//    centre and come apart for one that does not, and both are asserted where
//    they come apart.
// 6. **Decomposition invariance on a coarse lattice**: the same volume from
//    every block size against a whole-volume reference, on the bits, with a
//    population that is genuinely not all-true. This is the case the blocked
//    arms actually run.
// 7. **The declaration is checked**: the population is declared at the
//    *statistic's* reach — the lattice distance plus the element, not the
//    element alone — an image that does not hold `Bool` is refused by name, and
//    the op refuses to run without its operand at all.
// 8. **The constant algebra is honest**, which matters more here than usual: a
//    short circuit has not read the mask, so a declaration that depends on it
//    would skip a block into values it would not have computed.
//
// No assertion here is on wall-clock time.

use ndarray::{Array3, ArrayView3};

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::ArrayEnvironment;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInputs};
use blockflow::ops::sliding::{sliding_histogram_into, Domain, RankQuery};
use blockflow::ops::{
    masked_rank_filter_into_with, AdaptiveThresholdOp, ElementShape, EmptyPopulation,
    ExcludedCentre, LocalStatistic, LocalStatisticOp, Population, Rank, Sampling,
    StructuringElement, Total,
};
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

/// Deliberately not a multiple of the spacing on any axis, so the lattice's
/// unsampled margins differ from each other and from the block boundaries.
const VOLUME: [usize; 3] = [17, 13, 11];
/// Written by phase 0, read by phase 1 as each window's population.
const MASK: usize = 1;
/// The coarse lattice the blocked arms use. One axis at a different spacing so
/// a transposition would be visible.
const SPACING: [usize; 3] = [4, 4, 3];

// ------------------------------------------------------------- fixtures --

/// A value per voxel with no plateaus, so selecting a different rank selects a
/// different number and a wrong rank cannot hide behind a tie.
fn image() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 7 + j * 3 + k * 11) % 17) as f64
    })
}

/// A box that holds its own centre: the ordinary case, where an excluded centre
/// implies a window one voxel shorter.
fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

/// The element the two conditions come apart on: **six faces and nothing in the
/// middle**, so a centre can be in the population with an empty window and out
/// of it with a full one.
fn hollow() -> StructuringElement {
    StructuringElement::from_offsets([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ])
    .unwrap()
}

/// The percentile the reference's arm takes, through the convention that states
/// the rank against the surviving population — which is the convention a masked
/// window makes visible in the first place.
fn rank() -> Rank {
    Rank::ceiling_percentile(0.25).unwrap()
}

/// The populations this file asserts over, each named by what it is *for*.
///
/// Every one of them is checked below for being what it says: a population that
/// quietly kept everything would make the assertions pass for no reason.
fn populations() -> Vec<(&'static str, Array3<bool>)> {
    let image = image();
    vec![
        ("half the volume", image.mapv(|value| value > 8.0)),
        ("nothing at all", Array3::from_elem(image.raw_dim(), false)),
        (
            // Only the centres are denied, so every window keeps all its
            // neighbours and loses at most its own middle voxel.
            "only the centres of a lattice of voxels",
            Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| {
                !(i % 3 == 1 && j % 3 == 1 && k % 3 == 1)
            }),
        ),
        (
            // The complement: centres kept, everything else denied. A box
            // element then has a one-voxel window and a hollow one has none.
            "only the neighbours of a lattice of voxels",
            Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| {
                i % 3 == 1 && j % 3 == 1 && k % 3 == 1
            }),
        ),
        (
            "a slab, so whole windows fall out at once",
            Array3::from_shape_fn(image.raw_dim(), |(i, _, _)| !(5..12).contains(&i)),
        ),
    ]
}

/// The policies asserted over, each with the element it is interesting on.
fn policies() -> Vec<(&'static str, Population)> {
    vec![
        ("select at an excluded centre", Population::new()),
        (
            "select, carrying the centre where nothing survives",
            Population::new().carrying_the_centre(),
        ),
        (
            "fill at an excluded centre",
            Population::new().filling_excluded_centres(-1.0),
        ),
        (
            "fill, and carry where nothing survives",
            Population::new()
                .filling_excluded_centres(-1.0)
                .carrying_the_centre(),
        ),
    ]
}

/// The statistic under test at a spacing of one, which is the no-lattice case
/// and the case the dense filter is also defined on.
fn dense_statistic(element: StructuringElement) -> LocalStatistic {
    LocalStatistic::new(element, [1, 1, 1], blockflow::ops::Statistic::Rank(rank())).unwrap()
}

/// The same statistic on the coarse lattice.
fn sampled_statistic(element: StructuringElement) -> LocalStatistic {
    LocalStatistic::sampled(
        element,
        Sampling::Centred { spacing: SPACING },
        blockflow::ops::Statistic::Rank(rank()),
    )
    .unwrap()
}

/// Whole-volume, through the statistic's own entry point.
fn statistic_of(
    statistic: &LocalStatistic,
    image: &Array3<f64>,
    mask: Option<&Array3<bool>>,
    population: Population,
) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros(image.raw_dim());
    let at = Anchor::whole(VOLUME);
    match mask {
        None => statistic
            .evaluate_into(image.view(), &at, out.view_mut())
            .unwrap(),
        Some(mask) => statistic
            .evaluate_masked_into(image.view(), mask.view(), &at, population, out.view_mut())
            .unwrap(),
    }
    out
}

/// Whole-volume, through the **dense selection** — a different kernel in a
/// different module, reached by a different code path.
fn selection_of(
    image: &Array3<f64>,
    mask: &Array3<bool>,
    element: &StructuringElement,
    centre: ExcludedCentre<f64>,
) -> Array3<f64> {
    let ordered = image.mapv(Total);
    let mut out = Array3::from_elem(ordered.raw_dim(), Total(0.0));
    masked_rank_filter_into_with(
        ordered.view(),
        mask.view(),
        element,
        rank(),
        centre.map(Total),
        out.view_mut(),
    )
    .unwrap();
    out.mapv(|value| value.0)
}

/// Equality **on the bits**. `==` would let a last-bit difference through, and a
/// last-bit difference is still a different answer from the one the whole-volume
/// run would have given.
#[track_caller]
fn identical(got: &Array3<f64>, want: &Array3<f64>, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let differing = got
        .iter()
        .zip(want.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{what}: {differing} voxels differ on the bits"
    );
}

#[track_caller]
fn differs(got: &Array3<f64>, want: &Array3<f64>, what: &str) {
    let differing = got
        .iter()
        .zip(want.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(differing > 0, "{what}: the two agreed everywhere");
}

// -------------------------------- 0. the fixtures are what they say --

/// A mask that masks nothing tests nothing. Every population above is checked
/// for excluding *some* voxels and keeping *some*, except the one whose whole
/// job is to exclude everything.
#[test]
fn every_population_under_test_is_genuinely_partial() {
    for (name, mask) in populations() {
        let kept = mask.iter().filter(|&&keep| keep).count();
        let total = mask.len();
        if name == "nothing at all" {
            assert_eq!(kept, 0, "{name}");
            continue;
        }
        assert!(kept > 0 && kept < total, "{name}: kept {kept} of {total}");
    }
}

/// A box wide enough that its window runs off the volume's faces from the
/// **lattice's** own sample positions, which the radius-one element does not.
///
/// Truncation at a face is a case the truncation rule has to survive alongside
/// masking — both remove voxels from the population, and `Rank::resolve` sees
/// only the count that is left — so it needs a fixture where it happens.
fn wide() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [3, 3, 3])
}

/// The windows really are truncated, at a spacing of one and on the lattice.
/// Stated as a check because a later change to `VOLUME`, `SPACING` or either
/// element could quietly remove the case without failing anything else.
#[test]
fn windows_are_truncated_at_the_volumes_faces() {
    // At a spacing of one every voxel is a sample, so the voxel at zero has a
    // window running off the face on every axis.
    let (lo, _) = element().sides(0);
    assert!(lo > 0, "a radius-zero element truncates nowhere");

    // On the coarse lattice it takes the wider element.
    let lattice =
        blockflow::ops::SampleLattice::of(&Sampling::Centred { spacing: SPACING }, VOLUME).unwrap();
    let mut truncating = 0;
    for (axis, &extent) in VOLUME.iter().enumerate() {
        let (lo, hi) = wide().sides(axis);
        let first = lattice.centre(axis, 0);
        let last = lattice.centre(axis, lattice.count(axis) - 1);
        if first < lo || last + hi >= extent {
            truncating += 1;
        }
    }
    assert_eq!(
        truncating, 3,
        "every axis must have a sample window running off a face"
    );
}

// ----------------------------- 1. an all-true population is the plain form --

#[test]
fn a_population_that_keeps_everything_is_the_unmasked_statistic() {
    let image = image();
    let all = Array3::from_elem(image.raw_dim(), true);
    for statistic in [dense_statistic(element()), sampled_statistic(element())] {
        let plain = statistic_of(&statistic, &image, None, Population::new());
        for (name, population) in policies() {
            let masked = statistic_of(&statistic, &image, Some(&all), population);
            identical(&masked, &plain, name);
        }
    }
}

// ------------------------------- 2. the population changes the answer --

#[test]
fn the_population_changes_the_answer() {
    let image = image();
    for statistic in [dense_statistic(element()), sampled_statistic(element())] {
        let plain = statistic_of(&statistic, &image, None, Population::new());
        for (name, mask) in populations() {
            let masked = statistic_of(&statistic, &image, Some(&mask), Population::new());
            differs(&masked, &plain, name);
        }
    }
}

// ------------------- 3. byte-identity with the dense masked selection --

/// **The strongest check available.** At a spacing of one every voxel is its own
/// sample, `SampleLattice::bracket` gives the degenerate bracket, and `lerp`
/// returns its first argument exactly — so the lattice statistic of a
/// `Statistic::Rank` *is* the dense selection, and the two must agree on the
/// bits over every population and both policies.
#[test]
fn at_a_spacing_of_one_the_masked_statistic_is_the_dense_masked_selection() {
    let image = image();
    let mut checked = 0;
    for element in [element(), hollow()] {
        let statistic = dense_statistic(element.clone());
        for (population_name, mask) in populations() {
            for centre in [ExcludedCentre::Select, ExcludedCentre::Fill(-1.0)] {
                // The dense selection carries the centre where nothing
                // survives, so the statistic is asked to do the same. That is
                // the one place the two definitions had a choice, and
                // `EmptyPopulation` is where the choice is stated.
                let population = Population {
                    centre,
                    empty: EmptyPopulation::Centre,
                };
                let got = statistic_of(&statistic, &image, Some(&mask), population);
                let want = selection_of(&image, &mask, &element, centre);
                identical(
                    &got,
                    &want,
                    &format!(
                        "{population_name}, centre {centre:?}, element {}",
                        element.len()
                    ),
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 20, "two elements, five populations, two policies");
}

/// The other half of the same claim: under `EmptyPopulation::Reduce` the two
/// **must** differ, and only where the population emptied a window. Asserted so
/// that the agreement above is a property of the stated policy rather than of
/// the arithmetic being insensitive to it.
#[test]
fn the_empty_window_rule_is_the_only_place_the_two_definitions_part() {
    let image = image();
    let element = element();
    let statistic = dense_statistic(element.clone());
    // A population that empties some windows and not others.
    let mask = Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| {
        i % 3 == 1 && j % 3 == 1 && k % 3 == 1
    });

    let reduced = statistic_of(
        &statistic,
        &image,
        Some(&mask),
        Population {
            centre: ExcludedCentre::Select,
            empty: EmptyPopulation::Reduce,
        },
    );
    let carried = statistic_of(
        &statistic,
        &image,
        Some(&mask),
        Population {
            centre: ExcludedCentre::Select,
            empty: EmptyPopulation::Centre,
        },
    );
    let dense = selection_of(&image, &mask, &element, ExcludedCentre::Select);
    identical(&carried, &dense, "carrying the centre");
    differs(&reduced, &dense, "asking the statistic instead");

    // and the difference is exactly the emptied windows: a rank of nothing is
    // zero, which is what `Statistic::reduce_with` has always answered.
    let empties = Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| {
        !element.offsets().iter().any(|offset| {
            let at = [
                i as isize + offset[0],
                j as isize + offset[1],
                k as isize + offset[2],
            ];
            (0..3).all(|axis| at[axis] >= 0 && (at[axis] as usize) < VOLUME[axis])
                && mask[[at[0] as usize, at[1] as usize, at[2] as usize]]
        })
    });
    for index in ndarray::indices(image.raw_dim()) {
        let index = [index.0, index.1, index.2];
        if empties[index] {
            assert_eq!(
                reduced[index], 0.0,
                "an empty window reduces to zero at {index:?}"
            );
        } else {
            assert_eq!(
                reduced[index].to_bits(),
                dense[index].to_bits(),
                "a surviving window must agree at {index:?}"
            );
        }
    }
}

// --------------------- 4. byte-identity with the sliding histogram --

/// A third traversal of the same definition. The sliding form carries a
/// histogram along a scan line and never gathers a window at all, so agreement
/// on the bits is evidence about the *convention* rather than about one loop.
#[test]
fn the_masked_statistic_agrees_with_the_sliding_histogram_on_the_bits() {
    let image = image();
    let element = element();
    let statistic = dense_statistic(element.clone());
    let narrow = image.mapv(|value| value as u8);
    let domain = Domain::of::<u8>().unwrap();
    let query = RankQuery::new(rank(), &element);

    for (name, mask) in populations() {
        let mut sliding = Array3::<u8>::zeros(narrow.raw_dim());
        sliding_histogram_into(
            narrow.view(),
            Some(mask.view()),
            &element,
            domain,
            &query,
            ExcludedCentre::Select,
            sliding.view_mut(),
        )
        .unwrap();
        let got = statistic_of(
            &statistic,
            &image,
            Some(&mask),
            Population::new().carrying_the_centre(),
        );
        identical(&got, &sliding.mapv(f64::from), name);
    }
}

// ------------------------- 5. the two conditions stay distinct --

/// With an element that misses its own centre, "the centre is out of the
/// population" and "the window came out empty" are independent, and the two
/// parameters must land on the cases their names describe.
#[test]
fn an_excluded_centre_and_an_empty_window_are_asked_separately() {
    let image = image();
    let element = hollow();
    let statistic = dense_statistic(element.clone());

    // Centre denied, neighbours kept: the window is full and the centre's own
    // bit is the only thing a policy can see.
    let centre_only =
        Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| !(i == 8 && j == 6 && k == 5));
    let filled = statistic_of(
        &statistic,
        &image,
        Some(&centre_only),
        Population::new().filling_excluded_centres(-1.0),
    );
    let selected = statistic_of(&statistic, &image, Some(&centre_only), Population::new());
    assert_eq!(filled[[8, 6, 5]], -1.0, "a denied centre takes the fill");
    assert_ne!(
        selected[[8, 6, 5]],
        -1.0,
        "and under Select it is filtered anyway, from a window that is full"
    );

    // Neighbours denied, centre kept: the window is empty and the centre's bit
    // says nothing is wrong. A fill must **not** apply here.
    let neighbours_only =
        Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| (i, j, k) == (8, 6, 5));
    let filled = statistic_of(
        &statistic,
        &image,
        Some(&neighbours_only),
        Population::new()
            .filling_excluded_centres(-1.0)
            .carrying_the_centre(),
    );
    assert_eq!(
        filled[[8, 6, 5]].to_bits(),
        image[[8, 6, 5]].to_bits(),
        "an empty window is not an excluded centre, and the fill must not reach it"
    );
    // and where the centre is denied too, the fill does apply.
    assert_eq!(filled[[0, 0, 0]], -1.0);
}

// --------------------- 6. decomposition invariance on a lattice --

/// `image > 8`, into a `Bool` image. The population producer, kept here rather
/// than taken from `src/ops` because what is under test is the *consumer*.
struct Binarize;

impl BlockOp for Binarize {
    fn name(&self) -> &'static str {
        "binarize"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let source = input.view::<f64>()?;
        let mut out = out.view_mut::<bool>()?;
        ndarray::Zip::from(&mut out)
            .and(&source)
            .for_each(|slot, &value| *slot = value > 8.0);
        Ok(())
    }
}

fn masked_lattice_op(population: Population) -> LocalStatisticOp {
    masked_lattice_op_over(element(), population)
}

fn masked_lattice_op_over(element: StructuringElement, population: Population) -> LocalStatisticOp {
    LocalStatisticOp::new("masked-lattice-percentile", sampled_statistic(element))
        .masked_by(MASK)
        .with_population(population)
}

/// Phase 0 writes the population; phase 1 reads the image back through a source
/// leaf and evaluates the lattice statistic against it.
fn chain_over(element: StructuringElement, population: Population) -> Chain {
    Chain::sequence(vec![
        Chain::op(Binarize),
        Chain::source(0usize, Dtype::F64),
        Chain::op(masked_lattice_op_over(element, population)),
    ])
}

fn plan(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let reach = chain.reach3(&VOLUME);
    let phases = vec![
        PhaseDecomposition::derive(
            vec![0],
            vec![slots[0].display_name()],
            [0usize, 0, 0],
            [0usize, 0, 0],
            grid.clone(),
        ),
        PhaseDecomposition::derive(
            vec![1, 2],
            vec![slots[1].display_name(), slots[2].display_name()],
            reach,
            reach,
            grid.clone(),
        ),
    ];
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: reach,
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_images(chain).unwrap();
    plan
}

/// Block sizes that cut the lattice in every way that is available: not at all,
/// on one axis at three different offsets relative to the samples, on two, and
/// on all three.
fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0], 8).unwrap(),
        BlockGrid::along(VOLUME, &[0], 9).unwrap(),
        BlockGrid::along(VOLUME, &[1], 6).unwrap(),
        BlockGrid::along(VOLUME, &[2], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 2], 6).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 6).unwrap(),
    ]
}

fn run(grid: &BlockGrid, population: Population) -> Array3<f64> {
    run_over(element(), grid, population)
}

fn run_over(element: StructuringElement, grid: &BlockGrid, population: Population) -> Array3<f64> {
    let chain = chain_over(element, population);
    let decomposition = plan(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env =
        ArrayEnvironment::for_decomposition(image().into(), &decomposition, [4, 4, 4]).unwrap();
    execute(
        "masked-lattice",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<f64>().unwrap().to_owned()
}

#[test]
fn every_decomposition_gives_the_whole_volume_answer() {
    let image = image();
    let mask = image.mapv(|value| value > 8.0);
    // not vacuous, on both counts: the population is partial, and it changes
    // the answer this plan computes.
    let kept = mask.iter().filter(|&&keep| keep).count();
    assert!(
        kept > 0 && kept < mask.len(),
        "kept {kept} of {}",
        mask.len()
    );

    let mut checked = 0;
    for (name, population) in policies() {
        let statistic = sampled_statistic(element());
        let want = statistic_of(&statistic, &image, Some(&mask), population);
        differs(
            &want,
            &statistic_of(&statistic, &image, None, population),
            name,
        );
        for grid in grids() {
            identical(
                &run(&grid, population),
                &want,
                &format!("{name}, block {:?}", grid.block()),
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 32, "four policies over eight block grids");
}

/// The same claim where the sample windows are **truncated at the volume's
/// faces**, which is the case masking shares an arithmetic with: both remove
/// voxels from the population and `Rank::resolve` sees only what is left. A
/// block whose halo is clipped by a face must still produce the whole-volume
/// truncation and not its own.
#[test]
fn truncated_windows_at_the_faces_survive_every_decomposition() {
    let image = image();
    let mask = image.mapv(|value| value > 8.0);
    let statistic = sampled_statistic(wide());
    for population in [
        Population::new(),
        Population::new()
            .filling_excluded_centres(-1.0)
            .carrying_the_centre(),
    ] {
        let want = statistic_of(&statistic, &image, Some(&mask), population);
        differs(
            &want,
            &statistic_of(&statistic, &image, None, population),
            "the population must change a truncated window's answer too",
        );
        for grid in grids() {
            identical(
                &run_over(wide(), &grid, population),
                &want,
                &format!("wide element, block {:?}", grid.block()),
            );
        }
    }
}

/// The same, for the threshold that composes a statistic with a comparison:
/// masking decides what the level is computed *from*, and every voxel still gets
/// an answer.
#[test]
fn the_masked_threshold_is_decomposition_invariant_too() {
    let image = image();
    let mask = image.mapv(|value| value > 8.0);
    let op = AdaptiveThresholdOp::new("masked-threshold", sampled_statistic(element()), 1.0, 0.0)
        .masked_by(MASK);
    let plain = AdaptiveThresholdOp::new("threshold", sampled_statistic(element()), 1.0, 0.0);

    let chain = Chain::sequence(vec![
        Chain::op(Binarize),
        Chain::source(0usize, Dtype::F64),
        Chain::op(op),
    ]);
    let reach = chain.reach3(&VOLUME);
    let slots = chain.slots();

    let source: Voxels = image.clone().into();
    let mut whole = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let entries = [(MASK.into(), &Voxels::from(mask.clone()))];
    slots[2]
        .apply_with(
            &source,
            SourceInputs::new(&entries),
            &mut whole,
            &Anchor::whole(VOLUME),
        )
        .unwrap();
    let want = whole.view::<f64>().unwrap().to_owned();

    let mut unmasked = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    plain
        .apply(&source, &mut unmasked, &Anchor::whole(VOLUME))
        .unwrap();
    differs(
        &want,
        &unmasked.view::<f64>().unwrap().to_owned(),
        "the population must change the threshold",
    );

    for grid in grids() {
        let phases = vec![
            PhaseDecomposition::derive(
                vec![0],
                vec![slots[0].display_name()],
                [0usize, 0, 0],
                [0usize, 0, 0],
                grid.clone(),
            ),
            PhaseDecomposition::derive(
                vec![1, 2],
                vec![slots[1].display_name(), slots[2].display_name()],
                reach,
                reach,
                grid.clone(),
            ),
        ];
        let mut decomposition = Decomposition {
            volume: VOLUME,
            dtype: Dtype::F64,
            phases,
            chain_reach: reach,
        };
        decomposition.declare_dtypes(&chain).unwrap();
        decomposition.declare_source_images(&chain).unwrap();
        let workflow = Workflow::new(
            Chain::sequence(vec![
                Chain::op(Binarize),
                Chain::source(0usize, Dtype::F64),
                Chain::op(
                    AdaptiveThresholdOp::new(
                        "masked-threshold",
                        sampled_statistic(element()),
                        1.0,
                        0.0,
                    )
                    .masked_by(MASK),
                ),
            ]),
            VOLUME,
            Dtype::F64,
        );
        let env =
            ArrayEnvironment::for_decomposition(image.clone().into(), &decomposition, [4, 4, 4])
                .unwrap();
        execute(
            "masked-threshold",
            &workflow,
            &decomposition,
            &Hints::default(),
            &env,
        )
        .expect("a run");
        identical(
            &env.output().view::<f64>().unwrap().to_owned(),
            &want,
            &format!("block {:?}", grid.block()),
        );
    }
}

// ------------------------------------ 7. the declaration is checked --

/// **Not the element's reach**, which is what the dense filter declares. The
/// window sits around a lattice point, so the population is needed a lattice
/// distance further out — and it has to equal the op's own reach exactly, or
/// `check_source_images` refuses the plan.
#[test]
fn the_population_is_declared_at_the_statistics_reach_and_not_the_elements() {
    let op = masked_lattice_op(Population::new());
    let declared = op.source_inputs(VOLUME);
    assert_eq!(declared.len(), 1);
    assert_eq!(declared[0].image.index(), MASK);
    assert_eq!(
        declared[0].reach,
        op.statistic().reach_spec(VOLUME),
        "the population is consulted around every sample, so the statistic states both"
    );
    assert_ne!(
        declared[0].reach,
        element().reach_spec(),
        "declaring only the element would be short by a lattice distance everywhere"
    );
    for (axis, &extent) in VOLUME.iter().enumerate() {
        assert_eq!(
            op.reach(axis, extent),
            declared[0].reach.axis(axis).widest(extent),
            "axis {axis}: the operand must not want more than its phase fetches"
        );
    }
}

/// An unmasked op declares nothing, so nothing changed for a caller that never
/// asked for a population.
#[test]
fn an_unmasked_statistic_declares_no_operand() {
    let op = LocalStatisticOp::new("plain", sampled_statistic(element()));
    assert!(op.source_inputs(VOLUME).is_empty());
    assert_eq!(op.mask_image(), None);
}

#[test]
fn a_population_image_that_is_not_bool_is_refused_by_name() {
    let op = masked_lattice_op(Population::new());
    let input: Voxels = image().into();
    let wrong: Voxels = image().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let entries = [(MASK.into(), &wrong)];
    let failed = op
        .apply_with(
            &input,
            SourceInputs::new(&entries),
            &mut out,
            &Anchor::whole(VOLUME),
        )
        .unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("float64") && message.contains(&MASK.to_string()),
        "the refusal must name the image and what it holds: {message}"
    );
}

#[test]
fn the_op_refuses_to_run_without_its_population() {
    let op = masked_lattice_op(Population::new());
    let input: Voxels = image().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let failed = op
        .apply(&input, &mut out, &Anchor::whole(VOLUME))
        .unwrap_err();
    assert!(
        matches!(failed, Error::InvalidArgument(ref message) if message.contains("apply_with")),
        "an op with an operand must not have an answer without one: {failed}"
    );
}

/// A buffer may legitimately be asked for a sample it does not hold — the
/// samples bracketing a *buffer* extend a lattice distance past its edge, and
/// only halo voxels read them. That is not an error and must not become one: the
/// call goes through, and the block's core is unaffected because no core voxel
/// interpolates from a sample the buffer is missing.
///
/// The core's independence is what
/// [`every_decomposition_gives_the_whole_volume_answer`] asserts on the bits;
/// this asserts only that the call is allowed.
#[test]
fn a_sample_the_buffer_does_not_reach_is_not_an_error() {
    let statistic = sampled_statistic(element());
    let block = Array3::<f64>::zeros((2, 2, 2));
    let mask = Array3::from_elem(block.raw_dim(), true);
    let mut out = Array3::<f64>::zeros(block.raw_dim());
    let at = Anchor::new([0, 0, 0], VOLUME);
    statistic
        .evaluate_masked_into(
            block.view(),
            mask.view(),
            &at,
            Population::new()
                .filling_excluded_centres(0.0)
                .carrying_the_centre(),
            out.view_mut(),
        )
        .expect("a buffer short of a sample is a short halo, not a malformed call");
}

// -------------------------------- 8. the constant algebra is honest --

/// Every declaration is checked against **computing** the block, over a
/// population the short circuit cannot see. A wrong declaration here is a block
/// skipped into values it would not have produced.
#[test]
fn every_constant_declaration_agrees_with_computing_the_block() {
    // Three constants, chosen so that each policy's declaration survives for at
    // least one of them: zero is where a rank's empty answer and its constant
    // answer coincide, and minus one is the fill.
    let mut kept = 0;
    for constant in [3.5, 0.0, -1.0] {
        let image = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), constant);
        let masks = [
            Array3::from_elem(image.raw_dim(), true),
            Array3::from_elem(image.raw_dim(), false),
            Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| (i + j + k) % 2 == 0),
        ];
        for (name, population) in policies() {
            let op = masked_lattice_op(population);
            let Some(declared) = op.constant_maps_to(constant) else {
                continue;
            };
            kept += 1;
            for mask in &masks {
                let computed = statistic_of(
                    &sampled_statistic(element()),
                    &image,
                    Some(mask),
                    population,
                );
                assert!(
                    computed
                        .iter()
                        .all(|got| got.to_bits() == declared.to_bits()),
                    "{name} at {constant}: declared {declared} and computed something else"
                );
            }
        }
    }
    assert!(
        kept >= 4,
        "every policy withdrew everywhere, so nothing above was checked"
    );
}

/// And the declaration is **withdrawn** where it would have been a fact about
/// the population — which is the case that matters, because that is the one a
/// short circuit gets wrong silently.
#[test]
fn a_declaration_that_would_depend_on_the_population_is_withdrawn() {
    let carry = masked_lattice_op(Population::new().carrying_the_centre());
    assert_eq!(
        carry.constant_maps_to(3.5),
        Some(3.5),
        "carrying the centre of a constant block is the constant"
    );

    // A rank of an empty window is zero, and a rank of a constant window is the
    // constant, so under `Reduce` the two answers differ off zero.
    let reduce = masked_lattice_op(Population::new());
    assert_eq!(reduce.constant_maps_to(3.5), None);
    assert_eq!(reduce.constant_maps_to(0.0), Some(0.0));

    // A fill that is not the answer makes the answer a fact about the mask.
    let fill = masked_lattice_op(
        Population::new()
            .filling_excluded_centres(-1.0)
            .carrying_the_centre(),
    );
    assert_eq!(fill.constant_maps_to(3.5), None);
    assert_eq!(fill.constant_maps_to(-1.0), Some(-1.0));

    // An unmasked op is untouched: nothing about it became conditional.
    let plain = LocalStatisticOp::new("plain", sampled_statistic(element()));
    assert_eq!(plain.constant_maps_to(3.5), Some(3.5));
}

/// The masked path must not change what an unmasked op costs, and must charge
/// something for the second array it reads.
#[test]
fn reading_a_population_costs_more_than_not_reading_one() {
    let plain = LocalStatisticOp::new("plain", sampled_statistic(element()));
    let masked = masked_lattice_op(Population::new());
    assert!(
        masked.cost_per_voxel() > plain.cost_per_voxel(),
        "one byte of a second array per offset visited is not free"
    );
}

/// A helper the assertions above lean on: the population really is read at every
/// offset rather than at the sample centre only. Two masks that agree at every
/// lattice point and differ elsewhere must give different answers.
#[test]
fn the_population_is_consulted_at_every_offset_and_not_at_the_centre() {
    let image = image();
    let statistic = sampled_statistic(element());
    let lattice =
        blockflow::ops::SampleLattice::of(&Sampling::Centred { spacing: SPACING }, VOLUME).unwrap();
    let is_sample = |i: usize, axis: usize| lattice.positions(axis).contains(&i);
    let centres_only = Array3::from_shape_fn(image.raw_dim(), |(i, j, k)| {
        is_sample(i, 0) && is_sample(j, 1) && is_sample(k, 2)
    });
    let all = Array3::from_elem(image.raw_dim(), true);
    let a = statistic_of(&statistic, &image, Some(&centres_only), Population::new());
    let b = statistic_of(&statistic, &image, Some(&all), Population::new());
    differs(&a, &b, "a population read only at the centres would agree");
}

/// The kernel is generic over the buffer's element type and the view it is
/// handed need not be contiguous. Asserted because the masked gather indexes
/// two arrays rather than one.
#[test]
fn a_non_contiguous_view_gives_the_same_answer() {
    let image = image();
    let mask = image.mapv(|value| value > 8.0);
    let statistic = dense_statistic(element());
    let want = statistic_of(&statistic, &image, Some(&mask), Population::new());

    let padded = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2] * 2), |(i, j, k)| {
        if k % 2 == 0 {
            image[[i, j, k / 2]]
        } else {
            f64::NAN
        }
    });
    let padded_mask = Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2] * 2), |(i, j, k)| {
        k % 2 == 0 && mask[[i, j, k / 2]]
    });
    let view: ArrayView3<f64> = padded.slice(ndarray::s![.., .., ..;2]);
    let mask_view: ArrayView3<bool> = padded_mask.slice(ndarray::s![.., .., ..;2]);
    let mut out = Array3::<f64>::zeros(image.raw_dim());
    statistic
        .evaluate_masked_into(
            view,
            mask_view,
            &Anchor::whole(VOLUME),
            Population::new(),
            out.view_mut(),
        )
        .unwrap();
    identical(&out, &want, "a strided view");
}
