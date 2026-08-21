// SPDX-License-Identifier: MIT
//
// Original work for this crate. The **fixture data** in
// `ops::scikitimage_watershed::reference_case` is not — it is scikit-image
// reference output and lives beside the BSD-3-Clause translation whose header
// records where it came from. This file reads it and asserts against it.
//
// **A seeded watershed, and the one thing that makes it hard.**
//
// The claims here, in the order they depend on each other:
//
// 1. **The decisive case.** A `5 x 5 x 3` fixture on which this algorithm and a
//    plausible priority flood disagree about the *partition* rather than about a
//    boundary voxel: skimage assigns 32/9/10 voxels to three seeds where the
//    substitute written here assigns 6/26/12 and an earlier port of this op
//    assigned 25/9/13. Written first, because "a priority flood that looks
//    right" is the failure mode of this op and it passes every other test in
//    this file; the substitute is implemented rather than described, so the
//    fixture is *shown* to be decisive rather than claimed to be.
// 2. **Tie-breaking is the whole difficulty, and it is reproduced.** Priority is
//    `(cost, age)`; `age` is one global counter; ties on both keys fall to the
//    queue array's layout. Asserted three ways — a flat cost volume where
//    *nothing but* tie-breaking decides the answer and the split comes out
//    asymmetric; a half of the volume that no flood can reach, which changes the
//    answer in the other half anyway; and the key being the seed's *position*
//    rather than its name.
// 3. **A basin is not a connected component.** Measured, because the reason this
//    op exists rather than being served by `ops::detect` over `ops::components`
//    rests on the gap being large.
// 4. **The op is a planning barrier, and the planner acts on the declaration.**
// 5. **A blocked run really does differ, and by how much.** Run by hand, since
//    the planner will not offer such a plan, and counted.
// 6. **Ground truth.** `synthetic::Scene::object_table` supplies a per-object
//    voxel count computed from the placement rule, so basin sizes are checked
//    against what is *true* rather than against a second implementation.
// 7. **The memory claim is a measurement.** The module states 18 B/voxel dense
//    plus 32 B per queued item; the queue's peak is measured against its bound.
// 8. **The declarations hold**: dtypes refused by name, and no answer without
//    seeds.

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use ndarray::{s, Array3};

use blockflow::decomposition::{
    is_planning_barrier, splittable_axes, Constraints, Decomposition, PhaseDecomposition,
};
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInputs};
use blockflow::ops::scikitimage_watershed::reference_case;
use blockflow::ops::{
    label_regions_into, seeded_watershed, seeded_watershed_into_reporting_peak, SeededWatershedOp,
    Separation, WATERSHED_COST, WATERSHED_LINE_COST,
};
use blockflow::reach::{AxisReach, Reach};
use blockflow::strategy::{execute, Enumerating, Hints, Strategy, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;

/// Image 1 in the chains below: the seeds.
const SEEDS: usize = 1;
/// Image 2: the floodable region.
const MASK: usize = 2;

// ------------------------------------------------------------- fixtures --

fn reference_fixture() -> (Array3<f64>, Array3<u32>, Array3<bool>) {
    let shape = (
        reference_case::SHAPE[0],
        reference_case::SHAPE[1],
        reference_case::SHAPE[2],
    );
    let source = Array3::from_shape_vec(shape, reference_case::SOURCE.to_vec()).expect("5 x 5 x 3");
    let cost = source.mapv(|value| -value);
    let mask = source.mapv(|value| value > reference_case::THRESHOLD);
    let mut seeds = Array3::<u32>::zeros(source.raw_dim());
    for (label, at) in reference_case::SEEDS {
        seeds[at] = label;
    }
    (cost, seeds, mask)
}

fn sizes_of(labels: &Array3<u32>) -> BTreeMap<u32, usize> {
    let mut sizes = BTreeMap::new();
    for &label in labels {
        if label != 0 {
            *sizes.entry(label).or_insert(0usize) += 1;
        }
    }
    sizes
}

/// A volume with plenty of exact ties: values are integers in a small range, so
/// equal costs are everywhere and the order among them is the whole question.
fn tie_heavy(shape: [usize; 3], modulus: usize) -> Array3<f64> {
    Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        (((i * 37 + j * 101 + k * 17) ^ (i * j + k * 5)) % modulus) as f64
    })
}

// -------------------------------------------- 1. the substitute is wrong --

/// **A plausible priority flood.** A binary heap ordered on cost alone, a voxel
/// labelled when it is *pushed*, the neighbour's own cost kept as its priority,
/// and the separating line carved by clearing an already-labelled neighbour.
///
/// Every one of those four is a defensible reading of "seeded watershed" and the
/// result is a different partition. This exists so that the fixture below is
/// *shown* to separate the two rather than asserted to.
fn a_plausible_priority_flood(
    cost: &Array3<f64>,
    seeds: &Array3<u32>,
    mask: &Array3<bool>,
) -> Array3<u32> {
    #[derive(PartialEq)]
    struct Item(f64, usize);
    impl Eq for Item {}
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            // A max-heap of the negated cost is a min-heap of the cost.
            other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
        }
    }
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let shape = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let strides = [shape[1] * shape[2], shape[2], 1usize];
    let flat: Vec<f64> = cost.iter().copied().collect();
    let inside: Vec<bool> = mask.iter().copied().collect();
    let mut out: Vec<u32> = seeds
        .iter()
        .zip(&inside)
        .map(|(&label, &ok)| if ok { label } else { 0 })
        .collect();

    let mut heap = BinaryHeap::new();
    for (index, &label) in out.iter().enumerate() {
        if label != 0 {
            heap.push(Item(flat[index], index));
        }
    }
    let neighbours = |index: usize| {
        let mut coords = [0usize; 3];
        let mut rem = index;
        for axis in (0..3).rev() {
            coords[axis] = rem % shape[axis];
            rem /= shape[axis];
        }
        let mut found = Vec::with_capacity(6);
        for axis in 0..3 {
            if coords[axis] > 0 {
                found.push(index - strides[axis]);
            }
            if coords[axis] + 1 < shape[axis] {
                found.push(index + strides[axis]);
            }
        }
        found
    };

    while let Some(Item(_, index)) = heap.pop() {
        let label = out[index];
        if label == 0 {
            continue;
        }
        for neighbour in neighbours(index) {
            if !inside[neighbour] {
                continue;
            }
            match out[neighbour] {
                0 => {
                    out[neighbour] = label;
                    heap.push(Item(flat[neighbour], neighbour));
                }
                other if other != label => {
                    // "Carve the line by clearing the meeting voxel."
                    out[neighbour] = 0;
                }
                _ => {}
            }
        }
    }
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), out).expect("same shape")
}

/// The one case this whole op is built against.
#[test]
fn the_partition_is_skimages_and_not_a_plausible_floods() {
    let (cost, seeds, mask) = reference_fixture();

    let ours = seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(mask.view()),
        Separation::Line,
    )
    .unwrap();
    assert_eq!(
        ours.iter().copied().collect::<Vec<_>>(),
        reference_case::EXPECTED.to_vec(),
        "the reference partition, on bits"
    );

    let sizes = sizes_of(&ours);
    let ours_sizes: Vec<usize> = (1..=3).map(|label| sizes[&label]).collect();
    assert_eq!(ours_sizes, reference_case::SIZES.to_vec());

    let substitute = a_plausible_priority_flood(&cost, &seeds, &mask);
    let substitute_sizes = sizes_of(&substitute);
    let substitute_sizes: Vec<usize> = (1..=3)
        .map(|label| substitute_sizes.get(&label).copied().unwrap_or(0))
        .collect();
    println!(
        "5 x 5 x 3 reference fixture: skimage {ours_sizes:?}, a plausible flood \
         {substitute_sizes:?} (the earlier port gave {:?})",
        reference_case::NAIVE_SIZES
    );
    assert_ne!(
        ours_sizes, substitute_sizes,
        "if the substitute agreed here the fixture would pin nothing"
    );
}

// ------------------------------------------------------- 2. tie-breaking --

/// **A flat cost volume**: every voxel costs the same, so nothing but the
/// tie-breaking decides the partition. If the queue were stable in any way — by
/// insertion, by index, by label — the two seeds would split this symmetric
/// volume symmetrically. They do not.
#[test]
fn on_a_flat_cost_the_answer_is_entirely_the_tie_breaking() {
    let shape = (9usize, 9, 9);
    let cost = Array3::<f64>::zeros(shape);
    let mut seeds = Array3::<u32>::zeros(shape);
    seeds[[4, 4, 2]] = 1;
    seeds[[4, 4, 6]] = 2;

    let labels = seeded_watershed(cost.view(), seeds.view(), None, Separation::Adjacent).unwrap();
    let sizes = sizes_of(&labels);
    println!("flat cost, two symmetric seeds: {sizes:?}");
    assert_eq!(
        sizes.values().sum::<usize>(),
        9 * 9 * 9,
        "every voxel is labelled"
    );
    assert_ne!(
        sizes[&1], sizes[&2],
        "a symmetric split would mean the queue had a stable rule, and the whole \
         difficulty of this op is that it does not"
    );
}

/// **The barrier, demonstrated rather than argued.**
///
/// The mask here is two components with a two-voxel gap between them, so no
/// flood can cross from one to the other: whatever happens in the far half is
/// *geometrically* incapable of reaching the near half. Adding seeds over there
/// — and, separately, changing only the far half's cost — moves voxels here
/// anyway, because `age` is one counter over the whole volume and the far half's
/// pushes interleave with this half's, shifting every later tie.
///
/// That is the whole reason this op declares [`Reach::all`]. A halo cannot
/// express "everything that is happening elsewhere at the same cost".
#[test]
fn the_partition_is_a_function_of_the_whole_volume() {
    let shape = [24usize, 12, 8];
    let source = tie_heavy(shape, 7);
    let cost = source.mapv(|value| -value);
    let mut mask = Array3::from_elem((shape[0], shape[1], shape[2]), true);
    for j in 0..shape[1] {
        for k in 0..shape[2] {
            mask[[12, j, k]] = false;
            mask[[13, j, k]] = false;
        }
    }
    let near = s![0..12, .., ..];

    let seeds_of = |far: bool| {
        let mut seeds = Array3::<u32>::zeros((shape[0], shape[1], shape[2]));
        seeds[[1, 1, 1]] = 1;
        seeds[[10, 10, 6]] = 2;
        seeds[[5, 6, 3]] = 3;
        if far {
            seeds[[20, 6, 4]] = 4;
            seeds[[16, 2, 2]] = 5;
        }
        seeds
    };
    let flood = |seeds: &Array3<u32>, cost: &Array3<f64>| {
        seeded_watershed(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
        )
        .unwrap()
    };

    let alone = flood(&seeds_of(false), &cost);
    let with_far_seeds = flood(&seeds_of(true), &cost);
    let labelled = alone
        .slice(near)
        .iter()
        .filter(|&&label| label != 0)
        .count();
    let moved_by_seeds = alone
        .slice(near)
        .iter()
        .zip(with_far_seeds.slice(near).iter())
        .filter(|(left, right)| left != right)
        .count();

    // The same again, changing only the *cost* of the unreachable half.
    let mut far_cost = cost.clone();
    far_cost.slice_mut(s![14.., .., ..]).fill(-1.0);
    let with_far_cost = flood(&seeds_of(true), &far_cost);
    let moved_by_cost = with_far_seeds
        .slice(near)
        .iter()
        .zip(with_far_cost.slice(near).iter())
        .filter(|(left, right)| left != right)
        .count();

    println!(
        "an unreachable half of the volume: {moved_by_seeds} of {labelled} labelled voxels here \
         moved when seeds were added there, {moved_by_cost} when only its cost changed"
    );
    assert!(
        moved_by_seeds > 0,
        "nothing in the near half moved, so the queue is not globally coupled and this op \
         should not be declaring a barrier"
    );
    assert!(moved_by_cost > 0);
}

/// The other half of the tie-breaking story, and the one that is easy to get
/// backwards: the seeds are pushed in **raveled index order**, so their pop
/// order is a function of where they are and not of what they are called.
/// Renaming them permutes the output labels and moves no boundary.
#[test]
fn renaming_the_seeds_is_a_pure_relabelling() {
    let cost = tie_heavy([12, 12, 8], 5);
    let places = [[1usize, 1, 1], [10, 10, 6], [1, 10, 1], [10, 1, 6]];

    let mut seen = BTreeSet::new();
    for order in [[1u32, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]] {
        let mut seeds = Array3::<u32>::zeros(cost.raw_dim());
        for (label, at) in order.iter().zip(places) {
            seeds[at] = *label;
        }
        let labels = seeded_watershed(cost.view(), seeds.view(), None, Separation::Line).unwrap();
        // Compare the *partition*, not the labelling: map each voxel to the seed
        // position that owns it, so a permutation of the names is not a
        // difference.
        let by_place: Vec<usize> = labels
            .iter()
            .map(|&label| {
                order
                    .iter()
                    .position(|name| *name == label)
                    .map(|at| at + 1)
                    .unwrap_or(0)
            })
            .collect();
        seen.insert(by_place);
    }
    assert_eq!(
        seen.len(),
        1,
        "three namings of one set of seed positions gave {} partitions; the tie-break key is \
         the position, and if that stops being true every caller's labels become order-sensitive",
        seen.len()
    );

    // But moving one seed by a single voxel does move the far corner, which is
    // what makes the statement above a fact about the key rather than about the
    // fixture being insensitive.
    let mut here = Array3::<u32>::zeros(cost.raw_dim());
    for (index, at) in places.iter().enumerate() {
        here[*at] = index as u32 + 1;
    }
    let base = seeded_watershed(cost.view(), here.view(), None, Separation::Line).unwrap();
    let mut nudged = here.clone();
    nudged[places[3]] = 0;
    nudged[[10, 1, 5]] = 4;
    let nudged = seeded_watershed(cost.view(), nudged.view(), None, Separation::Line).unwrap();
    let far = s![0..4, 0..4, ..];
    let moved = base
        .slice(far)
        .iter()
        .zip(nudged.slice(far).iter())
        .filter(|(left, right)| left != right)
        .count();
    println!(
        "one seed moved one voxel: {moved} of {} voxels moved in the opposite corner",
        base.slice(far).len()
    );
    assert!(moved > 0, "the fixture is insensitive and proves nothing");
}

// --------------------------------------- 3. a basin is not a component --

/// **The trap this op exists to remove.** The separating line clears voxels
/// *between* labels, which routinely cuts one basin into several six-connected
/// pieces — so counting components of the labelled output is not a way to count
/// basins, and `ops::detect`'s `Moments::count` over those components is not a
/// basin size.
#[test]
fn a_basin_is_not_a_connected_component() {
    let shape = [32usize, 32, 24];
    let source = tie_heavy(shape, 13);
    let cost = source.mapv(|value| -value);
    let mask = source.mapv(|value| value > 2.0);
    let mut seeds = Array3::<u32>::zeros(source.raw_dim());
    let mut label = 0u32;
    for i in (2..shape[0]).step_by(5) {
        for j in (2..shape[1]).step_by(5) {
            for k in (2..shape[2]).step_by(5) {
                if mask[[i, j, k]] {
                    label += 1;
                    seeds[[i, j, k]] = label;
                }
            }
        }
    }

    let labels = seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(mask.view()),
        Separation::Line,
    )
    .unwrap();
    let basins = sizes_of(&labels).len();

    let labelled = labels.mapv(|label| label != 0);
    let mut components = Array3::<u32>::zeros(labelled.raw_dim());
    let component_count =
        label_regions_into(labelled.view(), components.view_mut()).unwrap() as usize;

    println!(
        "{shape:?}: {basins} basins, {component_count} six-connected components of the same \
         labelled voxels — a factor of {:.2}",
        component_count as f64 / basins as f64
    );
    assert!(
        component_count > basins,
        "the line split no basin, so this fixture demonstrates nothing"
    );
}

// ----------------------------------------------- 4. the barrier declared --

#[test]
fn the_op_declares_the_whole_volume_and_the_planner_reads_it() {
    let volume = [32usize, 16, 16];
    let op = SeededWatershedOp::new("watershed", SEEDS, Separation::Line).within(MASK);

    let reach = op.reach_spec(volume);
    assert_eq!(reach, Reach::all());
    for axis in 0..3 {
        assert_eq!(reach.axis(axis), &AxisReach::All);
        assert!(reach.is_whole_axis(axis, volume[axis]));
        assert_eq!(
            op.reach(axis, volume[axis]),
            volume[axis],
            "the symmetric bound must stay a bound on the spec"
        );
    }
    assert!(reach.is_barrier(volume));
    assert!(is_planning_barrier(
        &Chain::op(SeededWatershedOp::new("watershed", SEEDS, Separation::Line).within(MASK)),
        volume
    ));
    assert!(
        splittable_axes(&[0, 1, 2], &reach, volume).is_empty(),
        "no axis of a full-reach op may be cut"
    );

    // Both operands are declared over the whole volume too: a flood consults
    // every voxel of all three or it is a different flood.
    let declared = op.source_inputs(volume);
    assert_eq!(declared.len(), 2);
    assert_eq!(declared[0].image.index(), SEEDS);
    assert_eq!(declared[0].reach, Reach::all());
    assert_eq!(declared[1].image.index(), MASK);
    assert_eq!(declared[1].reach, Reach::all());
}

/// The planner, asked for a plan over a chain containing this op, collapses the
/// phase carrying it to one block — with no help from a special case.
#[test]
fn the_planner_gives_the_barrier_phase_a_single_block() {
    let volume = VOLUME;
    let workflow = Workflow::new(chain(), volume, Dtype::F64);
    // Candidates well below the volume, so a phase that *may* be cut is cut and
    // "one block" means the declaration was read rather than that nothing fits.
    let constraints = Constraints {
        block_candidates: vec![4, 8],
        split_axes: vec![0],
        ..Default::default()
    };
    let plan = Enumerating::default()
        .decompose(&workflow, &constraints)
        .expect("a plan");
    println!(
        "a chain ending in a seeded watershed decomposes into {} phase(s): {:?}",
        plan.n_phases(),
        plan.phases
            .iter()
            .map(|phase| (
                phase.names.clone(),
                phase.grid.block(),
                phase.grid.n_blocks()
            ))
            .collect::<Vec<_>>()
    );
    plan.check().expect("whatever it chose must tile");
    let flood = plan
        .phases
        .iter()
        .find(|phase| phase.names.iter().any(|name| name.contains("watershed")))
        .expect("the watershed is in some phase");
    assert_eq!(
        flood.grid.n_blocks(),
        1,
        "the phase carrying the watershed was cut into {} blocks",
        flood.grid.n_blocks()
    );
    assert!(
        plan.phases.iter().any(|phase| phase.grid.n_blocks() > 1),
        "no phase was cut at all, so 'the watershed's phase has one block' says nothing"
    );
    assert!(
        plan.n_phases() >= 2,
        "the chain fused across the barrier into {} phase(s)",
        plan.n_phases()
    );
}

// ----------------------------------------- 5. a blocked run really differs --

/// One block's worth of the flood, run on its own fetch region and cut back to
/// its core — which is what a decomposed run would be if the planner allowed it.
fn blocked_by_hand(
    cost: &Array3<f64>,
    seeds: &Array3<u32>,
    mask: &Array3<bool>,
    block: [usize; 3],
    halo: usize,
) -> Array3<u32> {
    let volume = [cost.shape()[0], cost.shape()[1], cost.shape()[2]];
    let mut out = Array3::<u32>::zeros(cost.raw_dim());
    let mut start = [0usize; 3];
    loop {
        let core_end = [
            (start[0] + block[0]).min(volume[0]),
            (start[1] + block[1]).min(volume[1]),
            (start[2] + block[2]).min(volume[2]),
        ];
        let read_start = [
            start[0].saturating_sub(halo),
            start[1].saturating_sub(halo),
            start[2].saturating_sub(halo),
        ];
        let read_end = [
            (core_end[0] + halo).min(volume[0]),
            (core_end[1] + halo).min(volume[1]),
            (core_end[2] + halo).min(volume[2]),
        ];
        let cut = s![
            read_start[0]..read_end[0],
            read_start[1]..read_end[1],
            read_start[2]..read_end[2]
        ];
        let labels = seeded_watershed(
            cost.slice(cut).to_owned().view(),
            seeds.slice(cut).to_owned().view(),
            Some(mask.slice(cut).to_owned().view()),
            Separation::Line,
        )
        .expect("a block is a volume");
        let inner = s![
            start[0] - read_start[0]..core_end[0] - read_start[0],
            start[1] - read_start[1]..core_end[1] - read_start[1],
            start[2] - read_start[2]..core_end[2] - read_start[2]
        ];
        out.slice_mut(s![
            start[0]..core_end[0],
            start[1]..core_end[1],
            start[2]..core_end[2]
        ])
        .assign(&labels.slice(inner));

        // Odometer over the grid.
        let mut axis = 0;
        loop {
            start[axis] += block[axis];
            if start[axis] < volume[axis] {
                break;
            }
            start[axis] = 0;
            axis += 1;
            if axis == 3 {
                return out;
            }
        }
    }
}

/// **The proof, not the argument.** The planner will not offer a blocked plan
/// for this op, so one is run by hand and the voxels that move are counted.
/// Reported at several block sizes and several halos, because "it differs" is
/// worth much less than "it differs by this much and a wider halo does not fix
/// it".
#[test]
fn a_blocked_run_is_not_the_whole_volume_answer() {
    let shape = [32usize, 32, 24];
    let source = tie_heavy(shape, 13);
    let cost = source.mapv(|value| -value);
    let mask = source.mapv(|value| value > 2.0);
    let mut seeds = Array3::<u32>::zeros(source.raw_dim());
    let mut label = 0u32;
    for i in (2..shape[0]).step_by(7) {
        for j in (2..shape[1]).step_by(7) {
            for k in (2..shape[2]).step_by(7) {
                if mask[[i, j, k]] {
                    label += 1;
                    seeds[[i, j, k]] = label;
                }
            }
        }
    }

    let whole = seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(mask.view()),
        Separation::Line,
    )
    .unwrap();
    let labelled = whole.iter().filter(|&&label| label != 0).count();

    let mut any_differed = false;
    for block in [[16usize, 32, 24], [16, 16, 24], [8, 8, 8]] {
        for halo in [1usize, 4, 12] {
            let blocked = blocked_by_hand(&cost, &seeds, &mask, block, halo);
            let moved = whole
                .iter()
                .zip(blocked.iter())
                .filter(|(left, right)| left != right)
                .count();
            println!(
                "block {block:?}, halo {halo}: {moved} of {labelled} labelled voxels differ \
                 ({:.1}%)",
                100.0 * moved as f64 / labelled as f64
            );
            any_differed |= moved > 0;
        }
    }
    assert!(
        any_differed,
        "no block size moved a voxel, so the barrier declaration is over-strict and this op \
         should be decomposed"
    );
}

/// The other half of the same claim: the **only** decomposition the planner will
/// admit — one block — reproduces the whole-volume answer bit for bit, through
/// the real executor.
#[test]
fn the_admissible_decomposition_is_byte_identical() {
    let expected = whole_volume_reference();
    let grid = BlockGrid::new(VOLUME, VOLUME).unwrap();
    let produced = run(&grid);
    assert_eq!(produced, expected);

    // And the grids a planner would offer for the *earlier* phases do not
    // change it either: only the flood's own phase is pinned to one block.
    for edge in [4usize, 8, 16] {
        let cut = BlockGrid::along(VOLUME, &[0], edge).unwrap();
        let produced = run_with_split_prefix(&cut);
        assert_eq!(
            produced, expected,
            "the phases before the barrier were cut at edge {edge} and the answer moved"
        );
    }
}

// ------------------------------------------------------- 6. ground truth --

/// The objects whose support has no six-neighbour belonging to a different
/// object — the ones for which "the basin is the object" is even a question the
/// data can answer. Placement is random, so which objects those are is a fact
/// about the scene rather than something a spec can promise.
fn objects_that_touch_nothing(labels: &Array3<u32>) -> BTreeSet<u32> {
    let shape = [labels.shape()[0], labels.shape()[1], labels.shape()[2]];
    let mut present = BTreeSet::new();
    let mut touching = BTreeSet::new();
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let here = labels[[i, j, k]];
                if here == 0 {
                    continue;
                }
                present.insert(here);
                for step in [[1usize, 0, 0], [0, 1, 0], [0, 0, 1]] {
                    let at = [i + step[0], j + step[1], k + step[2]];
                    if at[0] >= shape[0] || at[1] >= shape[1] || at[2] >= shape[2] {
                        continue;
                    }
                    let there = labels[at];
                    if there != 0 && there != here {
                        touching.insert(here);
                        touching.insert(there);
                    }
                }
            }
        }
    }
    present.difference(&touching).copied().collect()
}

/// **Against the placement rule, not against a second opinion.**
///
/// A clean scene has a hard edge and a zero background, so `intensity > 0` is
/// exactly the ground-truth label support and the cost *inside* an object is
/// flat — which means the flood within an object is decided by tie-breaking
/// alone, and any drift in the queue would show up here as a size that is off by
/// a few voxels rather than as a plausible-looking blob.
///
/// For an object that touches no other, the answer is not approximately right,
/// it is exactly `ObjectRecord::voxels`. Objects are placed at random and some
/// of them do touch; those are the subject of the next test, and mixing the two
/// would turn an exact claim into a tolerance.
#[test]
fn every_basin_is_exactly_its_object_when_objects_do_not_touch() {
    let scene = Scene::new(
        SceneSpec::clean([40, 40, 32], 7)
            .with_objects(24)
            .with_radius(3.0, 5.0),
    )
    .unwrap();
    let rendered = scene.render();
    let table = scene.object_table();
    let isolated = objects_that_touch_nothing(&rendered.labels);

    let cost = rendered.intensity.mapv(|value| -value);
    let mask = rendered.intensity.mapv(|value| value > 0.0);
    let mut seeds = Array3::<u32>::zeros(rendered.intensity.raw_dim());
    let mut seeded = 0usize;
    for record in &table {
        if record.voxels == 0 {
            continue;
        }
        let at = [
            record.centroid[0] as usize,
            record.centroid[1] as usize,
            record.centroid[2] as usize,
        ];
        assert!(
            mask[at],
            "object {} has its centroid outside its own support",
            record.id
        );
        seeds[at] = record.id;
        seeded += 1;
    }
    assert!(
        seeded >= 10,
        "only {seeded} objects — not much of a fixture"
    );
    assert!(
        isolated.len() >= 5,
        "only {} of {seeded} objects are isolated, which is not enough to assert on",
        isolated.len()
    );

    let labels = seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(mask.view()),
        Separation::Line,
    )
    .unwrap();
    let sizes = sizes_of(&labels);

    assert_eq!(
        sizes.len(),
        seeded,
        "every seed must win a basin and no seed may win two labels"
    );
    let mut checked = 0usize;
    for record in &table {
        if record.voxels == 0 || !isolated.contains(&record.id) {
            continue;
        }
        assert_eq!(
            sizes.get(&record.id).copied().unwrap_or(0) as u64,
            record.voxels,
            "basin {} against the ground-truth voxel count",
            record.id
        );
        // And not one voxel of it came from anywhere else.
        let leaked = labels
            .iter()
            .zip(rendered.labels.iter())
            .filter(|(&basin, &truth)| basin == record.id && truth != record.id)
            .count();
        assert_eq!(
            leaked, 0,
            "basin {} took {leaked} foreign voxels",
            record.id
        );
        checked += 1;
    }
    println!(
        "{seeded} objects placed, {checked} of them touching nothing: every one of those \
         basins is exactly its ground-truth object, to the voxel"
    );
}

/// The harder half, and the finding worth writing down.
///
/// When objects **interpenetrate**, ground truth gives the whole overlap to the
/// lower id while the flood splits it by cost, so the two answer different
/// questions and cannot agree everywhere. What is asserted is the part that is a
/// property of the partition rather than of the scene — no seed loses its basin,
/// so nothing merged, and each basin's plurality owner is its own object, so
/// nothing was captured wholesale — and the agreement is *reported*.
///
/// The number is lower than it looks like it should be, and the reason is worth
/// having: on a **flat plateau** — a hard-edged object of uniform brightness —
/// the brighter of two neighbours floods its own object at its own low cost
/// first, so its pushes into the dimmer neighbour carry smaller `age` values
/// than the dimmer seed's own pushes and win every tie there. A bright object
/// therefore eats into a dim one across a plateau. That is skimage's behaviour,
/// it is what "byte-identical to the reference" commits this op to, and it is a
/// reason for a caller to smooth a plateau rather than a reason to change this.
#[test]
fn on_touching_objects_the_partition_is_measured_against_ground_truth() {
    let scene = Scene::new(
        SceneSpec::new([40, 40, 32], 11)
            .with_objects(24)
            .with_radius(3.0, 6.0)
            .with_touching(0.6, 0.15),
    )
    .unwrap();
    let rendered = scene.render();
    let table = scene.object_table();
    let threshold = 0.3;

    let cost = rendered.intensity.mapv(|value| -value);
    let mask = rendered.intensity.mapv(|value| value > threshold);
    let mut seeds = Array3::<u32>::zeros(rendered.intensity.raw_dim());
    let mut seeded = 0usize;
    for record in &table {
        let at = [
            record.centroid[0] as usize,
            record.centroid[1] as usize,
            record.centroid[2] as usize,
        ];
        if record.voxels == 0 || !mask[at] {
            continue;
        }
        seeds[at] = record.id;
        seeded += 1;
    }

    let labels = seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(mask.view()),
        Separation::Line,
    )
    .unwrap();
    let sizes = sizes_of(&labels);
    let assigned: usize = sizes.values().sum();
    let agreed = labels
        .iter()
        .zip(rendered.labels.iter())
        .filter(|(&basin, &truth)| basin != 0 && basin == truth)
        .count();

    // What each basin is made of, by ground-truth id.
    let mut composition: BTreeMap<u32, BTreeMap<u32, usize>> = BTreeMap::new();
    for (&basin, &truth) in labels.iter().zip(rendered.labels.iter()) {
        if basin != 0 {
            *composition
                .entry(basin)
                .or_default()
                .entry(truth)
                .or_insert(0) += 1;
        }
    }
    let own_plurality = composition
        .iter()
        .filter(|(basin, parts)| {
            parts
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(owner, _)| owner == *basin)
                .unwrap_or(false)
        })
        .count();

    println!(
        "{seeded} interpenetrating objects at threshold {threshold}: {assigned} voxels \
         assigned, {agreed} of them ({:.1}%) carry the ground-truth id; {own_plurality} of \
         {} basins have their own object as plurality owner",
        100.0 * agreed as f64 / assigned as f64,
        composition.len()
    );
    assert_eq!(
        sizes.len(),
        seeded,
        "a seed that lost its basin means two objects merged"
    );
    assert!(
        own_plurality * 5 >= composition.len() * 4,
        "only {own_plurality} of {} basins are mostly their own object, which is a partition \
         that has stopped following the objects at all",
        composition.len()
    );
}

// ------------------------------------------------------------- 7. memory --

/// The module states 17 B/voxel of dense arrays plus 32 B per queued item, and
/// bounds the queue at six pushes per masked voxel. The bound is arithmetic; the
/// *peak* is not knowable without running, so it is measured — and the measured
/// figure is the interesting one, because on a densely masked volume the queue
/// is **larger than every dense array put together** and on a sparse one it
/// rounds to nothing. That is the whole of why the mask matters to a memory
/// budget, and the numbers this prints are the ones the module's table is built
/// from.
#[test]
fn the_queue_is_a_front_not_a_history() {
    let shape = [48usize, 48, 32];
    let source = tie_heavy(shape, 17);
    let cost = source.mapv(|value| -value);

    for threshold in [2.0f64, 8.0, 13.0] {
        let mask = source.mapv(|value| value > threshold);
        let masked = mask.iter().filter(|&&inside| inside).count();
        let mut seeds = Array3::<u32>::zeros(source.raw_dim());
        let mut label = 0u32;
        for i in (2..shape[0]).step_by(9) {
            for j in (2..shape[1]).step_by(9) {
                for k in (2..shape[2]).step_by(9) {
                    if mask[[i, j, k]] {
                        label += 1;
                        seeds[[i, j, k]] = label;
                    }
                }
            }
        }
        let mut out = Array3::<u32>::zeros(source.raw_dim());
        let peak = seeded_watershed_into_reporting_peak(
            cost.view(),
            seeds.view(),
            Some(mask.view()),
            Separation::Line,
            out.view_mut(),
        )
        .unwrap();

        let voxels = shape[0] * shape[1] * shape[2];
        let bound = 6 * masked;
        println!(
            "threshold {threshold}: {masked} of {voxels} voxels masked ({:.0}%), queue peaked \
             at {peak} items = {} KiB = {:.2} B per volume voxel, against a bound of {bound} \
             items = {} KiB ({:.2}% of the bound); the dense arrays are 18 B/voxel",
            100.0 * masked as f64 / voxels as f64,
            peak * 32 / 1024,
            (peak * 32) as f64 / voxels as f64,
            bound * 32 / 1024,
            100.0 * peak as f64 / bound as f64
        );
        assert!(
            peak <= bound,
            "the queue exceeded six pushes per masked voxel, so the stated bound is wrong"
        );
        assert!(peak > 0);
    }
}

// -------------------------------------------- 8. the declarations hold --

#[test]
fn the_op_refuses_to_run_without_its_seeds() {
    let op = SeededWatershedOp::new("watershed", SEEDS, Separation::Line);
    let input: Voxels = Array3::<f64>::zeros((4, 4, 4)).into();
    let mut out = Voxels::zeros(Dtype::U32, [4, 4, 4]).unwrap();
    let failed = op
        .apply(&input, &mut out, &Anchor::whole([4, 4, 4]))
        .unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("apply_with") && message.contains(&SEEDS.to_string()),
        "the refusal must name the image and the method: {message}"
    );
}

#[test]
fn a_seed_image_that_is_not_u32_is_refused_by_name() {
    let op = SeededWatershedOp::new("watershed", SEEDS, Separation::Line);
    let input: Voxels = Array3::<f64>::zeros((4, 4, 4)).into();
    let wrong: Voxels = Array3::<f64>::zeros((4, 4, 4)).into();
    let mut out = Voxels::zeros(Dtype::U32, [4, 4, 4]).unwrap();
    let entries = [(SEEDS.into(), &wrong)];
    let failed = op
        .apply_with(
            &input,
            SourceInputs::new(&entries),
            &mut out,
            &Anchor::whole([4, 4, 4]),
        )
        .unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("float64") && message.contains(&SEEDS.to_string()),
        "{message}"
    );
}

#[test]
fn a_mask_image_that_is_not_bool_is_refused_by_name() {
    let op = SeededWatershedOp::new("watershed", SEEDS, Separation::Line).within(MASK);
    let input: Voxels = Array3::<f64>::zeros((4, 4, 4)).into();
    let seeds: Voxels = Array3::<u32>::zeros((4, 4, 4)).into();
    let wrong: Voxels = Array3::<f64>::zeros((4, 4, 4)).into();
    let mut out = Voxels::zeros(Dtype::U32, [4, 4, 4]).unwrap();
    let entries = [(SEEDS.into(), &seeds), (MASK.into(), &wrong)];
    let failed = op
        .apply_with(
            &input,
            SourceInputs::new(&entries),
            &mut out,
            &Anchor::whole([4, 4, 4]),
        )
        .unwrap_err();
    let message = failed.to_string();
    assert!(
        message.contains("float64") && message.contains(&MASK.to_string()),
        "{message}"
    );
}

/// The cost is a **function of the separation**, because the two are different
/// amounts of queue traffic rather than the same flood with a different
/// finishing step: without a line a voxel is queued once, with one it is queued
/// again by every neighbour that pops before it. Measured at 4.7x; declared, so
/// the planner is not told one figure for two programs.
#[test]
fn the_declared_cost_follows_the_separation() {
    let line = SeededWatershedOp::new("watershed", SEEDS, Separation::Line);
    let adjacent = SeededWatershedOp::new("watershed", SEEDS, Separation::Adjacent);
    assert_eq!(line.cost_per_voxel(), WATERSHED_LINE_COST);
    assert_eq!(adjacent.cost_per_voxel(), WATERSHED_COST);
    assert!(
        line.cost_per_voxel() > adjacent.cost_per_voxel() * 2.0,
        "the two were priced the same, and they are not the same amount of work"
    );
}

#[test]
fn the_op_writes_labels_whatever_it_reads() {
    let op = SeededWatershedOp::new("watershed", SEEDS, Separation::Line);
    assert!(op.accepts(Dtype::F64));
    assert!(!op.accepts(Dtype::U16));
    assert_eq!(op.produces(Dtype::F64), Dtype::U32);
    assert_eq!(
        op.constant_maps_to(0.0),
        None,
        "a flat cost is exactly the case decided by tie-breaking alone, so there is no \
         constant to declare"
    );
}

// ------------------------------------------------- the executed workflow --

const VOLUME: [usize; 3] = [24, 16, 16];

/// The intensity the whole workflow is built on.
fn workflow_image() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        (((i * 31 + j * 17 + k * 7) % 19) as f64) / 19.0
    })
}

const WORKFLOW_THRESHOLD: f64 = 0.35;

/// Seeds on a regular lattice, so the seed volume is a function of position and
/// a block sees exactly the seeds inside it.
struct SeedLattice;

impl BlockOp for SeedLattice {
    fn name(&self) -> &'static str {
        "seeds"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::F64
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<(), blockflow::Error> {
        let source = input.view::<f64>()?;
        let mut out = out.view_mut::<u32>()?;
        for i in 0..source.shape()[0] {
            for j in 0..source.shape()[1] {
                for k in 0..source.shape()[2] {
                    let global = [at.offset[0] + i, at.offset[1] + j, at.offset[2] + k];
                    // A label that is a function of the global position, so a
                    // block and the whole volume agree on it.
                    let on_lattice = global[0] % 8 == 4 && global[1] % 8 == 4 && global[2] % 8 == 4;
                    out[[i, j, k]] = if on_lattice && source[[i, j, k]] > WORKFLOW_THRESHOLD {
                        (1 + (global[0] / 8) * 4 + (global[1] / 8) * 2 + global[2] / 8) as u32
                    } else {
                        0
                    };
                }
            }
        }
        Ok(())
    }
}

struct Binarize;

impl BlockOp for Binarize {
    fn name(&self) -> &'static str {
        "mask"
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

    fn apply(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<(), blockflow::Error> {
        let source = input.view::<f64>()?;
        let mut out = out.view_mut::<bool>()?;
        ndarray::Zip::from(&mut out)
            .and(&source)
            .for_each(|slot, &value| *slot = value > WORKFLOW_THRESHOLD);
        Ok(())
    }
}

/// Negate, so the flood runs toward the bright structure. Kept as its own op
/// rather than folded into the watershed: the cost volume is the caller's, and
/// this is what "the caller builds it" looks like.
struct Negate;

impl BlockOp for Negate {
    fn name(&self) -> &'static str {
        "negate"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn apply(
        &self,
        input: &Voxels,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<(), blockflow::Error> {
        let source = input.view::<f64>()?;
        let mut out = out.view_mut::<f64>()?;
        ndarray::Zip::from(&mut out)
            .and(&source)
            .for_each(|slot, &value| *slot = -value);
        Ok(())
    }
}

/// Image 0 is the image, 1 the seeds, 2 the mask, 3 the negated cost, 4 the
/// labels.
fn chain() -> Chain {
    Chain::sequence(vec![
        Chain::op(SeedLattice),
        Chain::source(0usize, Dtype::F64),
        Chain::op(Binarize),
        Chain::source(0usize, Dtype::F64),
        Chain::op(Negate),
        Chain::op(SeededWatershedOp::new("watershed", SEEDS, Separation::Line).within(MASK)),
    ])
}

/// The whole-volume answer, from the free function, built exactly the way the
/// chain builds it — so a disagreement is the framework and not two recipes.
fn whole_volume_reference() -> Array3<u32> {
    let image = workflow_image();
    let mut seeds = Array3::<u32>::zeros(image.raw_dim());
    for i in 0..VOLUME[0] {
        for j in 0..VOLUME[1] {
            for k in 0..VOLUME[2] {
                if i % 8 == 4 && j % 8 == 4 && k % 8 == 4 && image[[i, j, k]] > WORKFLOW_THRESHOLD {
                    seeds[[i, j, k]] = (1 + (i / 8) * 4 + (j / 8) * 2 + k / 8) as u32;
                }
            }
        }
    }
    let mask = image.mapv(|value| value > WORKFLOW_THRESHOLD);
    let cost = image.mapv(|value| -value);
    seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(mask.view()),
        Separation::Line,
    )
    .unwrap()
}

fn plan_with(prefix: &BlockGrid) -> (Chain, Decomposition) {
    let chain = chain();
    let slots = chain.slots();
    let whole = BlockGrid::new(VOLUME, VOLUME).unwrap();
    let name = |slot: usize| slots[slot].display_name();
    let phases = vec![
        // seeds
        PhaseDecomposition::derive(
            vec![0],
            vec![name(0)],
            [0usize, 0, 0],
            [0usize, 0, 0],
            prefix.clone(),
        ),
        // the image again, then the mask
        PhaseDecomposition::derive(
            vec![1, 2],
            vec![name(1), name(2)],
            [0usize, 0, 0],
            [0usize, 0, 0],
            prefix.clone(),
        ),
        // the image again, then the cost
        PhaseDecomposition::derive(
            vec![3, 4],
            vec![name(3), name(4)],
            [0usize, 0, 0],
            [0usize, 0, 0],
            prefix.clone(),
        ),
        // the flood: one block, because it declares the whole volume
        PhaseDecomposition::derive(vec![5], vec![name(5)], Reach::all(), Reach::all(), whole),
    ];
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [VOLUME[0], VOLUME[1], VOLUME[2]],
    };
    plan.declare_dtypes(&chain).unwrap();
    plan.declare_source_images(&chain).unwrap();
    (chain, plan)
}

fn run_with_split_prefix(prefix: &BlockGrid) -> Array3<u32> {
    let (chain, decomposition) = plan_with(prefix);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env =
        ArrayEnvironment::for_decomposition(workflow_image().into(), &decomposition, [4, 4, 4])
            .unwrap();
    execute(
        "seeded-watershed",
        &workflow,
        &decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    env.output().view::<u32>().unwrap().to_owned()
}

fn run(grid: &BlockGrid) -> Array3<u32> {
    run_with_split_prefix(grid)
}
