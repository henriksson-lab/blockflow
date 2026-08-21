// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A diamond whose second arm is a stored image.**
//
// A phase used to read exactly one image: `run_task` did `env.read(task.phase,
// fetch)` and nothing else. An op needing a second array therefore had to hold
// it — `ops::voxelwise::CombineOp` keeps an `Arc<Voxels>` of the whole volume
// and slices it at the anchor. That is correct, properly anchored, and one full
// copy of the array resident for the length of the run, at the sizes an
// out-of-core framework exists for.
//
// `Chain::Source` is a leaf that *reads* an image instead of computing one, so
// the second arm of a `Chain::Parallel` can be an array on disk. What this file
// asserts, in the order the claims depend on each other:
//
// 1. **It is the same answer.** Byte-identical to the same diamond with
//    `CombineOp` holding the array, across seven decompositions.
// 2. **The residency difference is the point.** The held form's operand is one
//    whole volume, always; the source form's peak is bounded by blocks, and the
//    image it reads is fetched once per block rather than once per run.
// 3. **Two readers keep an image alive.** The image read by the source leaf is
//    *not* freed when its immediate reader finishes, and *is* freed after the
//    last one — sampled during the run, not inferred from the end state.
// 4. **A forward reference is refused by name**, at plan time, before any block
//    runs.
// 5. **The halo guard still fires** for a phase with a source leaf, and so do
//    the guards on the image's extent and element type.
// 6. **Decomposition invariance**, which is the property the whole crate is
//    arranged around and which a second input is the obvious way to break.
//
// No assertion here is on wall-clock time.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use ndarray::Array3;

use blockflow::assemble::ImageId;
use blockflow::decomposition::{check_source_images, Decomposition, PhaseDecomposition};
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::{
    CombineOp, ElementShape, Logic, LogicCombine, Morphology, MorphologyOp, StructuringElement,
    VoxelwiseMapOp,
};
use blockflow::strategy::{execute, execute_observed, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Event, EventListener};

const VOLUME: [usize; 3] = [16, 12, 10];
/// The image the second arm reads: written by phase 0, read by phase 1 as its
/// input and by phase 2 through a source leaf. An *intermediate*, deliberately —
/// image 0 would have been the easy case, because nothing ever frees it.
const STORED: usize = 1;

// ------------------------------------------------------------- fixtures --

/// A sparse mask of small separated boxes — about 7% of the volume.
///
/// **Clustered rather than scattered, deliberately.** A pattern with a voxel
/// every few positions dilates to the whole volume and then erodes back to it,
/// which would make every arm of the diamond below the same array and every
/// comparison in this file true for no reason. Separated boxes keep the
/// dilation, the erosion and the input three genuinely different sets.
fn input() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        f64::from(i % 6 < 2 && j % 5 < 2 && k % 4 < 2)
    })
}

fn element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Box, [1, 1, 1])
}

/// Slot 0: a real neighbourhood op, so the phase that writes the stored image
/// has a halo and the source leaf is not being read beside a trivial chain.
fn dilate() -> Chain {
    Chain::op(MorphologyOp::new("dilate", Morphology::Dilate, element()))
}

/// Slot 1: an erosion, so image 2 is a strict subset of image 1 and `XOR`
/// against image 1 is a set difference rather than a constant.
fn erode() -> Chain {
    Chain::op(MorphologyOp::new("erode", Morphology::Erode, element()))
}

/// The arm of the diamond that *is* computed, kept an identity so the answer is
/// exactly `xor(image 2, image 1)` and nothing else.
fn computed_arm() -> Chain {
    Chain::op(VoxelwiseMapOp::new("keep", |value| value))
}

/// The chain under test: the third slot is a fan-in whose second branch reads a
/// image instead of computing one.
///
/// **`XOR` rather than `AND`, and the choice is load-bearing.** `AND` and `OR`
/// are idempotent, so a diamond joining an image with itself would give the same
/// answer as one joining it with the image beside it — and the test that the
/// arm reads the image it *names* would pass for an implementation that quietly
/// handed it the phase's own input. `XOR` distinguishes them: against itself it
/// is zero everywhere, against image 1 it is the set difference.
fn source_chain() -> Chain {
    Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::parallel(
            vec![computed_arm(), Chain::source(STORED, Dtype::F64)],
            Box::new(LogicCombine::new("xor", Logic::Xor)),
        )
        .unwrap(),
    ])
}

/// The same computation with the second operand held whole, which is what this
/// crate could express before. `stored` must be the entire volume; `CombineOp`
/// checks that against the anchor on every block.
fn held_chain(stored: Arc<Voxels>) -> Chain {
    Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::op(CombineOp::new("xor", Logic::Xor, stored)),
    ])
}

/// The whole-volume value of the image the source leaf reads: one dilation of
/// the input, computed in one call.
///
/// This is the array the held form has to keep, and computing it here is what
/// makes the comparison honest — the reference pays the cost the source form
/// avoids, rather than being handed something for free.
fn stored_image() -> Voxels {
    whole_volume(&dilate())
}

/// Apply `chain` to the input in one call, over the whole volume.
///
/// The reference every comparison here is against, and it is the same kernels
/// the blocked run uses — so a disagreement is a decomposition bug rather than
/// two implementations of the same idea drifting.
fn whole_volume(chain: &Chain) -> Voxels {
    let source: Voxels = input().into();
    let mut out = Voxels::zeros(
        chain.produces(Dtype::F64).unwrap(),
        chain.output_shape(VOLUME).unwrap(),
    )
    .unwrap();
    chain
        .apply(&source, &mut out, &blockflow::op::Anchor::whole(VOLUME))
        .unwrap();
    out
}

/// The two arms of the diamond, whole: image 1 and image 2.
fn arms() -> (Array3<f64>, Array3<f64>) {
    let one = whole_volume(&dilate());
    let two = whole_volume(&Chain::sequence(vec![dilate(), erode()]));
    (
        one.view::<f64>().unwrap().to_owned(),
        two.view::<f64>().unwrap().to_owned(),
    )
}

/// One phase per slot, so every stage really does materialise an image and the
/// stored image is an image rather than a buffer inside a phase.
fn one_phase_per_slot(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let reaches = [[1usize, 1, 1], [1, 1, 1], [0, 0, 0]];
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                reaches[slot],
                reaches[slot],
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [2, 2, 2],
    };
    plan.declare_dtypes(chain).unwrap();
    plan.declare_source_images(chain).unwrap();
    plan
}

fn grids() -> Vec<BlockGrid> {
    vec![
        BlockGrid::new(VOLUME, VOLUME).unwrap(),
        BlockGrid::along(VOLUME, &[0], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0], 8).unwrap(),
        BlockGrid::along(VOLUME, &[1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[2], 5).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1], 4).unwrap(),
        BlockGrid::along(VOLUME, &[0, 1, 2], 4).unwrap(),
    ]
}

fn run(chain: Chain, grid: &BlockGrid, hints: &Hints) -> (Array3<f64>, ArrayEnvironment) {
    let plan = one_phase_per_slot(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap();
    execute("source", &workflow, &plan, hints, &env).expect("a run");
    let out = env.output().view::<f64>().unwrap().to_owned();
    (out, env)
}

// ------------------------------------------------------- 1. same answer --

/// The property the whole change has to earn: reading the second arm a block at
/// a time is the same arithmetic as holding it.
#[test]
fn a_stored_second_arm_is_byte_identical_to_the_same_array_held_in_memory() {
    let stored = Arc::new(stored_image());
    for grid in grids() {
        let (sourced, _) = run(source_chain(), &grid, &Hints::default());
        let (held, _) = run(held_chain(Arc::clone(&stored)), &grid, &Hints::default());
        assert_eq!(sourced, held, "block {:?}", grid.block());
    }
}

/// Decomposition invariance, stated against the *whole-volume* answer rather
/// than against another decomposition — so a bug that moved every block the same
/// way would still be caught.
#[test]
fn every_decomposition_of_the_source_form_gives_the_whole_volume_answer() {
    let reference = {
        let whole = BlockGrid::new(VOLUME, VOLUME).unwrap();
        run(source_chain(), &whole, &Hints::default()).0
    };
    for grid in grids() {
        let (out, _) = run(source_chain(), &grid, &Hints::default());
        assert_eq!(out, reference, "block {:?}", grid.block());
    }
    // and it is *the* answer, not two runs agreeing on nothing: `XOR` over
    // set-ness against the two arms computed whole.
    let (one, two) = arms();
    let mut wanted = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (index, value) in wanted.indexed_iter_mut() {
        *value = f64::from((two[index] != 0.0) != (one[index] != 0.0));
    }
    assert_eq!(reference, wanted);
    // and the answer is not a constant, which would make every comparison above
    // true for the wrong reason
    assert!(wanted.iter().any(|&v| v == 0.0));
    assert!(wanted.iter().any(|&v| v == 1.0));
}

/// The arm really is a second array and not a copy of the first: replacing the
/// stored image with the phase's own input image changes the answer.
///
/// Without this, every assertion above would still pass for an implementation
/// that quietly handed the source leaf the buffer it was already given.
#[test]
fn the_arm_reads_the_image_it_names_and_not_the_one_the_phase_was_handed() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let (from_image_one, _) = run(source_chain(), &grid, &Hints::default());

    let own_input = Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::parallel(
            // Image 2 is what this phase is handed, so this arm duplicates the
            // other one and `XOR` folds to zero everywhere.
            vec![computed_arm(), Chain::source(2, Dtype::F64)],
            Box::new(LogicCombine::new("xor", Logic::Xor)),
        )
        .unwrap(),
    ]);
    let (from_image_two, _) = run(own_input, &grid, &Hints::default());
    assert!(from_image_two.iter().all(|&value| value == 0.0));
    assert_ne!(from_image_one, from_image_two);
}

// ------------------------------------------------------- 2. residency --

/// **The point of the exercise, as a number.**
///
/// The held form's second operand is one whole volume, held for the length of
/// the run, whatever the block size — and held *outside* the environment, so it
/// is a cost nothing in this crate can chunk, page out, free or even count.
///
/// The source form's second operand is an image. What it holds at once is
/// therefore what `EnvCounters` says it holds: peak resident bytes, from the
/// same counters `tests/image_lifetime.rs` reads image residency out of. That
/// stays below one whole array and falls as the blocks get smaller, because the
/// arm is fetched a block at a time and released with the first.
#[test]
fn the_source_form_never_holds_the_second_array_whole() {
    let whole_volume_bytes =
        (VOLUME[0] * VOLUME[1] * VOLUME[2] * std::mem::size_of::<f64>()) as u64;
    // What the held form costs, before a single block runs.
    let stored = Arc::new(stored_image());
    assert_eq!(stored.bytes(), whole_volume_bytes);

    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let (_, env) = run(source_chain(), &grid, &Hints::default());
    let peak = env.counters().peak_resident_bytes.load(Ordering::SeqCst);
    assert!(
        peak < whole_volume_bytes,
        "peak {peak} against a whole array of {whole_volume_bytes}"
    );

    // and it is the *blocks* that bound it: halving them halves the peak's
    // order, which a run holding the array whole could not do.
    let fine = BlockGrid::along(VOLUME, &[0, 1, 2], 4).unwrap();
    let (_, finer) = run(source_chain(), &fine, &Hints::default());
    let fine_peak = finer.counters().peak_resident_bytes.load(Ordering::SeqCst);
    assert!(fine_peak < peak, "{fine_peak} against {peak}");

    // and it costs no extra *image*, which is the residency `image_lifetime.rs`
    // counts: the same two survive the run either way. The whole difference is
    // that the held form's copy is one the framework never saw.
    let (_, held) = run(held_chain(Arc::clone(&stored)), &grid, &Hints::default());
    assert_eq!(env.resident_images(), held.resident_images());
}

/// The second arm is *read*, once per block, and the plan says so in advance.
///
/// `exact_read_voxels` is compared against the run to the voxel elsewhere in
/// this crate; a phase reading two images has to predict two images' worth or
/// that comparison silently starts measuring the wrong plan.
#[test]
fn the_plan_predicts_the_second_read_and_the_run_performs_it() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let chain = source_chain();
    let plan = one_phase_per_slot(&chain, &grid);
    let blocks = plan.phases[2].blocks.len();

    let predicted = plan.exact_read_voxels();
    let fetched = |phase: usize| -> usize {
        plan.phases[phase]
            .blocks
            .iter()
            .map(|block| block.source.voxels())
            .sum()
    };
    assert_eq!(
        predicted[2],
        2 * fetched(2),
        "one read of each of two images"
    );
    assert_eq!(
        predicted[1],
        fetched(1),
        "a phase with no source leaf reads one"
    );

    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap();
    let log = Arc::new(blockflow::ExecutionLog::new());
    let listeners: Vec<Arc<dyn EventListener>> = vec![log.clone()];
    let stats = execute_observed(
        "source",
        &workflow,
        &plan,
        &Hints::default(),
        &env,
        &listeners,
    )
    .unwrap();
    assert_eq!(predicted.iter().sum::<usize>() as u64, stats.read_voxels);

    // one fetch of the stored image per block of the phase that reads it, which
    // is what "a block at a time" means as a count
    let reads = log
        .events()
        .iter()
        .filter(|event| matches!(event, Event::RegionRead { image, .. } if *image == STORED))
        .count();
    assert_eq!(
        reads,
        blocks + plan.phases[1].blocks.len(),
        "phase 1 reads it as its input, phase 2 reads it as its second arm"
    );
}

// ------------------------------------------------- 3. two readers --

/// Who reads what, straight off the plan. This is the refcount, and it is what
/// replaced "an image is read by exactly one phase".
#[test]
fn an_image_read_by_a_source_leaf_has_two_readers() {
    let chain = source_chain();
    let plan = one_phase_per_slot(&chain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    assert_eq!(plan.phases[2].source_images, vec![STORED]);

    assert_eq!(plan.readers_of_image(0), vec![0]);
    assert_eq!(plan.readers_of_image(STORED), vec![1, 2]);
    assert_eq!(plan.readers_of_image(2), vec![2]);
    assert_eq!(plan.readers_of_image(3), Vec::<usize>::new(), "the output");

    // so nothing dies when phase 1 ends — image 1 is still wanted — and both
    // intermediates die when phase 2 does
    assert_eq!(plan.images_dead_after(0), vec![0], "the input, never freed");
    assert_eq!(plan.images_dead_after(1), Vec::<usize>::new());
    assert_eq!(plan.images_dead_after(2), vec![STORED, 2]);

    // and with no source leaf it is exactly the rule it generalises
    let plain = Chain::sequence(vec![dilate(), erode(), computed_arm()]);
    let plain = one_phase_per_slot(&plain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    for phase in 0..plain.n_phases() {
        assert_eq!(plain.images_dead_after(phase), vec![phase], "phase {phase}");
    }
}

/// Sampled **during the run**, because the end state cannot tell the two rules
/// apart: under either one, everything internal is gone by the time `execute`
/// returns. What differs is *when*.
#[test]
fn the_stored_image_survives_its_first_reader_and_dies_after_its_last() {
    /// Records whether the stored image was still resident at the moment each
    /// phase finished — `Materialised` is emitted immediately after the
    /// executor has made its discard decision for that phase.
    struct Watch {
        env: Arc<ArrayEnvironment>,
        seen: Mutex<Vec<(usize, bool)>>,
    }

    impl EventListener for Watch {
        fn on_event(&self, event: &Event) {
            if let Event::Materialised { phase, .. } = event {
                self.seen
                    .lock()
                    .unwrap()
                    .push((*phase, self.env.is_discarded(STORED)));
            }
        }
    }

    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let chain = source_chain();
    let plan = one_phase_per_slot(&chain, &grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env =
        Arc::new(ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap());
    let watch = Arc::new(Watch {
        env: Arc::clone(&env),
        seen: Mutex::new(Vec::new()),
    });
    let listeners: Vec<Arc<dyn EventListener>> = vec![watch.clone()];
    execute_observed(
        "source",
        &workflow,
        &plan,
        &Hints::default(),
        &*env,
        &listeners,
    )
    .unwrap();

    let seen = watch.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![(0, false), (1, false), (2, true)],
        "image {STORED} is alive while phase 2 still needs it"
    );

    // The same chain without the source leaf frees it one phase earlier, which
    // is the measurement that says the refcount did something.
    let plain = Chain::sequence(vec![dilate(), erode(), computed_arm()]);
    let plan = one_phase_per_slot(&plain, &grid);
    let workflow = Workflow::new(plain, VOLUME, Dtype::F64);
    let env =
        Arc::new(ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap());
    let watch = Arc::new(Watch {
        env: Arc::clone(&env),
        seen: Mutex::new(Vec::new()),
    });
    let listeners: Vec<Arc<dyn EventListener>> = vec![watch.clone()];
    execute_observed(
        "plain",
        &workflow,
        &plan,
        &Hints::default(),
        &*env,
        &listeners,
    )
    .unwrap();
    assert_eq!(
        watch.seen.lock().unwrap().clone(),
        vec![(0, false), (1, true), (2, true)],
        "with one reader it dies as soon as that reader is done"
    );
}

/// A freed image must be loud, and the source form must not be the thing that
/// makes it quiet: pinning still works, and reading a freed image still fails.
#[test]
fn pinning_the_stored_image_keeps_it_and_the_answer_is_unchanged() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let (freed, freed_env) = run(source_chain(), &grid, &Hints::default());
    let hints = Hints {
        keep_images: [ImageId::from(STORED)].into_iter().collect(),
        ..Hints::default()
    };
    let (kept, kept_env) = run(source_chain(), &grid, &hints);
    assert_eq!(freed, kept);
    assert!(freed_env.is_discarded(STORED));
    assert!(!kept_env.is_discarded(STORED));
    assert_eq!(kept_env.resident_images(), freed_env.resident_images() + 1);
}

// -------------------------------------------------------- 4. the guards --

/// **Refused by name, at plan time.** A phase cannot read an image a later phase
/// writes, and the message has to say which two phases they are — discovering it
/// at the first block would mean discovering it after `prepare`, after the graph,
/// and once per block.
#[test]
fn a_forward_reference_is_refused_when_the_plan_is_made() {
    let chain = Chain::sequence(vec![
        dilate(),
        Chain::parallel(
            // Image 3 is written by phase 2 and this is phase 1.
            vec![computed_arm(), Chain::source(3, Dtype::F64)],
            Box::new(LogicCombine::new("and", Logic::And)),
        )
        .unwrap(),
        erode(),
    ]);
    let plan = one_phase_per_slot(&chain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    let message = check_source_images(&chain, &plan).unwrap_err().to_string();
    assert!(message.contains("phase 1"), "{message}");
    assert!(message.contains("image 3"), "{message}");
    assert!(message.contains("phase 2"), "{message}");

    // and the executor refuses it too, before any block runs
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap();
    let failed = execute("source", &workflow, &plan, &Hints::default(), &env).unwrap_err();
    assert!(failed.to_string().contains("image 3"), "{failed}");
    assert_eq!(
        env.counters().ops_applied.load(Ordering::SeqCst),
        0,
        "refused before the first block"
    );
}

/// An image that does not exist at all.
#[test]
fn an_image_past_the_end_of_the_plan_is_refused() {
    let chain = Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::parallel(
            vec![computed_arm(), Chain::source(9, Dtype::F64)],
            Box::new(LogicCombine::new("and", Logic::And)),
        )
        .unwrap(),
    ]);
    let plan = one_phase_per_slot(&chain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    let message = check_source_images(&chain, &plan).unwrap_err().to_string();
    assert!(message.contains("image 9"), "{message}");
    assert!(message.contains("4 image(s)"), "{message}");
}

/// The declared element type is the image's, and a leaf that says otherwise is
/// caught by the plan rather than by a view that fails on the first block.
#[test]
fn a_source_leaf_that_misdeclares_the_element_type_is_refused() {
    // A bare leaf as the third slot: a phase that reads image 1 and writes it
    // on. The declaration is the only thing saying what it holds, which is
    // exactly the case the check exists for.
    let chain = Chain::sequence(vec![dilate(), erode(), Chain::source(STORED, Dtype::Bool)]);
    let plan = one_phase_per_slot(&chain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    let message = check_source_images(&chain, &plan).unwrap_err().to_string();
    assert!(message.contains("bool"), "{message}");
    assert!(message.contains("float64"), "{message}");
}

/// A plan whose record disagrees with its chain reads one image and prices
/// another, which is exactly what a parity-visible field must not be able to do.
#[test]
fn a_plan_whose_recorded_images_disagree_with_its_chain_is_refused() {
    let chain = source_chain();
    let mut plan = one_phase_per_slot(&chain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    plan.phases[2].source_images.clear();
    let message = check_source_images(&chain, &plan).unwrap_err().to_string();
    assert!(message.contains("[1]"), "{message}");
}

/// **The halo guard still fires**, on the phase that has the source leaf: a
/// source arm is reach 0, so it does not widen the halo, but it must not
/// suppress the check either.
#[test]
fn the_halo_guard_fires_for_a_phase_with_a_source_leaf() {
    // A fan-in whose computed arm has a real reach, so the phase holding the
    // source leaf is the one with a halo to get wrong.
    let chain = Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::parallel(
            vec![
                Chain::op(MorphologyOp::new("erode", Morphology::Erode, element())),
                Chain::source(STORED, Dtype::F64),
            ],
            Box::new(LogicCombine::new("and", Logic::And)),
        )
        .unwrap(),
    ]);
    let slots = chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let reaches = [[1usize, 1, 1], [0, 0, 0], [1, 1, 1]];
    // The third phase asks for a reach of one and is granted a halo of none.
    let halos = [[1usize, 1, 1], [0, 0, 0], [0, 0, 0]];
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                reaches[slot],
                halos[slot],
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [2, 2, 2],
    };
    plan.declare_source_images(&chain).unwrap();

    let message = plan.check().unwrap_err().to_string();
    assert!(message.contains("phase 2"), "{message}");
    assert!(message.contains("tile"), "{message}");

    // and the fan-in's reach really is the computed arm's, unchanged by the
    // arm that reads
    assert_eq!(slots[2].reach(0, VOLUME[0]), 1);
}

/// Reach 0, stated as the fold rather than as a claim: adding a source arm to a
/// diamond changes no plan's halo.
#[test]
fn a_source_arm_adds_nothing_to_the_reach() {
    let with_source = Chain::parallel(
        vec![
            Chain::op(MorphologyOp::new("dilate", Morphology::Dilate, element())),
            Chain::source(STORED, Dtype::F64),
        ],
        Box::new(LogicCombine::new("or", Logic::Or)),
    )
    .unwrap();
    let alone = Chain::op(MorphologyOp::new("dilate", Morphology::Dilate, element()));
    for (axis, &extent) in VOLUME.iter().enumerate() {
        assert_eq!(
            with_source.reach(axis, extent),
            alone.reach(axis, extent),
            "axis {axis}"
        );
    }
    assert_eq!(
        with_source.reach_spec(VOLUME).unwrap(),
        alone.reach_spec(VOLUME).unwrap()
    );
    // and it costs no compute, only a read
    assert_eq!(
        Chain::source(STORED, Dtype::F64).cost_per_voxel(),
        0.0,
        "a read is priced as bytes, not as work"
    );
}

/// Applied without its operand, a source leaf fails and says which image it
/// wanted — rather than producing a well-formed volume combined against
/// nothing.
#[test]
fn a_source_leaf_applied_with_no_operand_says_which_image_it_wanted() {
    let leaf = Chain::source(7, Dtype::F64);
    let source: Voxels = input().into();
    let mut out = Voxels::zeros(Dtype::F64, VOLUME).unwrap();
    let message = leaf
        .apply(&source, &mut out, &blockflow::op::Anchor::whole(VOLUME))
        .unwrap_err()
        .to_string();
    assert!(message.contains("image 7"), "{message}");
}

// --------------------------------------------------- 5. the fingerprint --

/// The image an arm reads changes voxels, so it changes the fingerprint — and a
/// plan that reads no second image fingerprints exactly as it did before source
/// leaves existed.
#[test]
fn the_fingerprint_records_which_image_is_read_and_is_unchanged_without_one() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let plain = Chain::sequence(vec![dilate(), erode(), computed_arm()]);
    let plain_plan = one_phase_per_slot(&plain, &grid);

    let sourced = one_phase_per_slot(&source_chain(), &grid);
    // Only the recorded images differ; the geometry is identical.
    let mut stripped = sourced.clone();
    stripped.phases[2].source_images.clear();
    stripped.phases[2].names = plain_plan.phases[2].names.clone();
    assert_eq!(stripped.fingerprint(), plain_plan.fingerprint());
    assert_ne!(sourced.fingerprint(), stripped.fingerprint());

    let mut other = sourced.clone();
    other.phases[2].source_images = vec![0];
    assert_ne!(sourced.fingerprint(), other.fingerprint());
}

// ---------------------------------------------------- 6. under workers --

/// Concurrency changes nothing, which is the property the second read has to
/// not break: two workers fetching the same stored region is a shared *read*,
/// and a shared read has no hazard.
#[test]
fn several_workers_produce_the_same_answer() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let reference = run(source_chain(), &grid, &Hints::default()).0;
    for concurrency in [1usize, 2, 4] {
        let hints = Hints {
            concurrency,
            ..Hints::default()
        };
        let (out, _) = run(source_chain(), &grid, &hints);
        assert_eq!(out, reference, "{concurrency} workers");
    }
}

/// Two source leaves naming the same image are handed one buffer, and two
/// naming different images are handed the right one each.
#[test]
fn several_source_leaves_in_one_phase_each_get_the_image_they_named() {
    let chain = Chain::sequence(vec![
        dilate(),
        erode(),
        Chain::parallel(
            vec![
                computed_arm(),
                Chain::source(0, Dtype::F64),
                Chain::source(STORED, Dtype::F64),
                Chain::source(STORED, Dtype::F64),
            ],
            Box::new(LogicCombine::new("and", Logic::And)),
        )
        .unwrap(),
    ]);
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let plan = one_phase_per_slot(&chain, &grid);
    assert_eq!(
        plan.phases[2].source_images,
        vec![0, STORED],
        "one entry per image, not one per leaf"
    );

    let (out, _) = run(chain, &grid, &Hints::default());
    // `AND` over: image 2, image 0, then image 1 twice.
    let (one, two) = arms();
    let raw = input();
    let mut wanted = Array3::zeros((VOLUME[0], VOLUME[1], VOLUME[2]));
    for (index, value) in wanted.indexed_iter_mut() {
        *value = f64::from(two[index] != 0.0 && raw[index] != 0.0 && one[index] != 0.0);
    }
    assert_eq!(out, wanted);
    assert!(wanted.iter().any(|&v| v == 0.0));
    assert!(wanted.iter().any(|&v| v == 1.0));
}
