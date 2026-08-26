// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// An image is written by one phase, read by exactly one phase, and then dead.
// Until now nothing acted on that: every image of an `N`-phase plan was
// allocated at full volume and held for the whole run, so a twenty-stage chain
// kept twenty-one copies of the data resident while at most two of them were
// live. This suite is the evidence that it no longer does, and the evidence that
// freeing them changed no answer.
//
// Three properties, and the third is the one that makes the first two worth
// having:
//
// 1. **The saving is real and is a number.** Residency after a long chain is
//    counted, not described.
// 2. **The output is unchanged.** Byte-identical to the same chain run with
//    every image pinned, which is the old behaviour exactly.
// 3. **Reading a freed image fails.** An image that came back as zeros would be
//    the same class of defect as an unwritten image reading as zeros, which this
//    crate fills with NaN precisely so that it cannot happen quietly.

use std::sync::{Arc, LazyLock, Mutex};

use ndarray::Array3;

use blockflow::assemble::ImageId;
use blockflow::decomposition::{Decomposition, PhaseDecomposition, Visibility};
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::fragment::{
    append_fragment_phase, BlockOutput, BlockView, FragmentInput, FragmentOp, FragmentOutput,
    PhaseWork,
};
use blockflow::geometry::BlockGrid;
use blockflow::listener::EventListener;
use blockflow::log::Event;
use blockflow::op::Chain;
use blockflow::ops::{ElementShape, Morphology, MorphologyOp, StructuringElement, VoxelwiseMapOp};
use blockflow::probes::BlockSummaryOp;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute, execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [16, 12, 10];
const STAGES: usize = 20;

fn input() -> Array3<f64> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        ((i * 31 + j * 17 + k * 7) % 11) as f64
    })
}

/// A chain long enough that holding every image is obviously the wrong thing:
/// twenty stages, each a cheap voxelwise map so the test measures residency
/// rather than arithmetic.
fn long_chain() -> Chain {
    Chain::sequence(
        (0..STAGES)
            .map(|stage| {
                let bias = stage as f64;
                Chain::op(VoxelwiseMapOp::new("step", move |value| value + bias * 0.0))
            })
            .collect(),
    )
}

/// One phase per slot, so every stage really does materialise an image.
fn one_phase_per_slot(chain: &Chain) -> Decomposition {
    let slots = chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [0, 0, 0],
                [0, 0, 0],
                grid.clone(),
            )
        })
        .collect();
    Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [0, 0, 0],
    }
}

fn run(hints: &Hints) -> (Array3<f64>, ArrayEnvironment) {
    let chain = long_chain();
    let plan = one_phase_per_slot(&chain);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env = ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap();
    execute("images", &workflow, &plan, hints, &env).expect("a run");
    let out = env.output().view::<f64>().unwrap().to_owned();
    (out, env)
}

/// Everything pinned: the behaviour before this existed, kept as the reference.
fn keep_everything() -> Hints {
    Hints {
        keep_images: (0..=STAGES).map(ImageId::from).collect(),
        ..Hints::default()
    }
}

/// End of run: the input and the output, and nothing else.
///
/// Note what this is *not* measuring. Peak residency during the run is one
/// higher — while phase `p` runs it holds image `p` and image `p+1`, plus image
/// 0, which is never freed — so the bound is three images at any moment against
/// `N + 1` before. What is asserted here is the end state, because it is the one
/// a test can read without sampling the run.
#[test]
fn a_long_chain_ends_holding_two_images_instead_of_all_of_them() {
    let (_, env) = run(&Hints::default());
    assert_eq!(env.resident_images(), 2, "the input and the output");

    // and the old behaviour, for the comparison to be against something real
    let (_, kept) = run(&keep_everything());
    assert_eq!(kept.resident_images(), STAGES + 1);
}

#[test]
fn freeing_the_intermediates_changes_no_voxel() {
    let (freed, _) = run(&Hints::default());
    let (kept, _) = run(&keep_everything());
    assert_eq!(freed, kept);
    // and it is the answer, not merely two runs agreeing
    assert_eq!(freed, input());
}

/// Image 0 is somebody else's array and the output is what the run is for.
/// Neither is ever a candidate, whatever the hints say.
#[test]
fn the_input_and_the_output_are_never_freed() {
    let chain = long_chain();
    let plan = one_phase_per_slot(&chain);
    assert_eq!(plan.image_visibility(0), Visibility::Published);
    assert_eq!(
        plan.image_visibility(plan.n_images() - 1),
        Visibility::Published
    );
    for image in 1..plan.n_images() - 1 {
        assert_eq!(
            plan.image_visibility(image),
            Visibility::Internal,
            "image {image}"
        );
    }

    let (_, env) = run(&Hints::default());
    assert!(!env.is_discarded(0));
    assert!(!env.is_discarded(STAGES));
}

/// A freed image must be **loud**, not empty. Silence here would be the same
/// defect as an unwritten image reading as zeros.
#[test]
fn reading_a_freed_image_fails_and_says_why() {
    let (_, env) = run(&Hints::default());
    assert!(env.is_discarded(1));

    let region = blockflow::region::Region::new(&[0, 0, 0], &[4, 4, 4]);
    let message = env.read(1, &region).unwrap_err().to_string();
    assert!(message.contains("discarded"), "{message}");
    assert!(message.contains("keep_images"), "{message}");
}

/// Pinning one image keeps exactly that one.
#[test]
fn pinning_an_image_keeps_it_and_nothing_else() {
    let hints = Hints {
        keep_images: [ImageId::from(7)].into_iter().collect(),
        ..Hints::default()
    };
    let (out, env) = run(&hints);
    assert_eq!(out, input());
    assert!(!env.is_discarded(7));
    assert!(env.is_discarded(6));
    assert!(env.is_discarded(8));
    assert_eq!(
        env.resident_images(),
        3,
        "the input, the output, and the pinned one"
    );
}

/// The saving is in *bytes*, and a chain that changes element type makes the
/// point sharper than a uniform one: an intermediate held at eight bytes a voxel
/// costs eight times a mask image, and holding twenty of them is what this
/// stopped doing.
#[test]
fn the_saving_is_proportional_to_the_chain_length() {
    let per_image = (VOLUME[0] * VOLUME[1] * VOLUME[2] * std::mem::size_of::<f64>()) as u64;
    let (_, freed) = run(&Hints::default());
    let (_, kept) = run(&keep_everything());

    let freed_bytes = freed.resident_images() as u64 * per_image;
    let kept_bytes = kept.resident_images() as u64 * per_image;
    assert_eq!(kept_bytes / freed_bytes, 10, "21 images against 2");
    assert!(kept_bytes - freed_bytes > 0);
}

/// An op with a real reach, so the chain is not only voxelwise maps — the
/// lifetime rule must not depend on a phase reading exactly its own core.
#[test]
fn a_chain_with_halos_frees_its_intermediates_too() {
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    let chain = Chain::sequence(
        (0..4)
            .map(|_| {
                Chain::op(MorphologyOp::new(
                    "dilate",
                    Morphology::Dilate,
                    element.clone(),
                ))
            })
            .collect(),
    );
    let slots = chain.slots();
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [1, 1, 1],
                [1, 1, 1],
                grid.clone(),
            )
        })
        .collect();
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [4, 4, 4],
    };
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);

    let mut mask = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), 0.0);
    mask[[8, 6, 5]] = 1.0;
    let source: Voxels = mask.into();

    let env = ArrayEnvironment::for_decomposition(source.clone(), &plan, [4, 4, 4]).unwrap();
    execute("halos", &workflow, &plan, &Hints::default(), &env).expect("a run");
    let freed = env.output().view::<f64>().unwrap().to_owned();
    assert_eq!(env.resident_images(), 2);

    let kept_env = ArrayEnvironment::for_decomposition(source, &plan, [4, 4, 4]).unwrap();
    let hints = Hints {
        keep_images: (0..=4).map(ImageId::from).collect(),
        ..Hints::default()
    };
    execute("halos", &workflow, &plan, &hints, &kept_env).expect("a run");
    assert_eq!(freed, kept_env.output().view::<f64>().unwrap().to_owned());
    assert_eq!(kept_env.resident_images(), 5);
}

// ------------------------------ the `readers_of_image` over-count --------
//
// `Decomposition::readers_of_image` counts phase `p` as a reader of image `p`
// unconditionally. A fragment phase declaring `reads_pixels() == false`
// performs no pixel IO at all — the executor issues no `Environment::read` for
// it — so for such a phase the count is false, and the image it is credited
// with reading is held for one phase longer than anything wants it.
//
// `peak_image_bytes` already works around this and says so
// (`decomposition.rs`, *Why this does not call `images_dead_after`*). What is
// below is the measurement that workaround did not have: the residency the
// executor actually holds, sampled during the run, against a negative control
// that is the same program with one thing changed.

/// Renders the stream into an image without reading one: `reads_pixels` is
/// `false` and `writes_pixels` is `true`. This is what makes the plan below end
/// in a written output rather than in a slot nothing fills.
struct RenderOp(&'static str, usize);

impl FragmentOp for RenderOp {
    fn name(&self) -> &'static str {
        "render"
    }

    /// Nothing crosses as pixels — none are read. The stream it consumes is
    /// its own block's entry, at fragment reach zero.
    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn writes_pixels(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(self.0, self.1)]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        Vec::new()
    }

    fn apply(&self, at: &BlockView<'_>) -> blockflow::error::Result<BlockOutput> {
        Ok(BlockOutput::nothing().with_pixels(at.output_buffer(0.0)?))
    }
}

/// The tallies of the long plan, and the one the short plan uses.
///
/// One per phase, because two phases may not write the same stream name; they
/// read nothing from each other, which is what makes each of them a phase that
/// performs no pixel IO for its own reasons rather than by inheritance.
/// The tally and its negative control are now **one probe with one bool
/// flipped**, which is what the paragraph above claims and what two
/// hand-written structs could only promise: `probes::BlockSummaryOp` reads its
/// block's pixels unless `with_pixels(false)` says otherwise, and that
/// declaration is the entire difference between the two plans below.
///
/// `LazyLock` because the probe owns its stream name and a `String` cannot be
/// built in a `static` initialiser. The plans want `&'static dyn FragmentOp`,
/// and a `static LazyLock` still gives one.
static TALLIES: LazyLock<[BlockSummaryOp; 3]> = LazyLock::new(|| {
    ["tally-a", "tally-b", "tally-c"].map(|stream| {
        BlockSummaryOp::new("tally", stream, Lifecycle::DeleteOnExit).with_pixels(false)
    })
});
static READING_TALLY: LazyLock<BlockSummaryOp> =
    LazyLock::new(|| BlockSummaryOp::new("reading-tally", "tally", Lifecycle::DeleteOnExit));

/// The pixel phase every plan below starts from: one voxelwise map, writing
/// image 1.
fn pixel_head() -> (Decomposition, Chain) {
    let chain = Chain::op(VoxelwiseMapOp::new("step", |value| value));
    let grid = BlockGrid::along(VOLUME, &[0], 8).unwrap();
    let plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases: vec![PhaseDecomposition::derive(
            vec![0],
            vec!["step".to_string()],
            [0, 0, 0],
            [0, 0, 0],
            grid,
        )],
        chain_reach: [0, 0, 0],
    };
    (plan, chain)
}

/// `tallies` fragment phases after the pixel phase, then one that renders the
/// last of their streams into the output image.
///
/// Image 1 is written by phase 0 and — when phase 1 is a tally with
/// `with_pixels(false)` — read by **nobody**: it does no pixel IO and no later
/// phase names image 1 in `source_images`. With the same probe left reading in
/// that slot it is genuinely read by phase 1, and that is the only difference
/// between the two plans — one bool, now, rather than two structs that had to
/// be kept identical by hand.
fn tally_plan(reads: bool, tallies: usize) -> (Decomposition, Chain, Vec<PhaseWork<'static>>) {
    let (mut plan, chain) = pixel_head();
    let mut work: Vec<PhaseWork<'static>> = vec![PhaseWork::Pixels];
    for index in 0..tallies {
        let op: &'static dyn FragmentOp = if reads && index == 0 {
            &*READING_TALLY
        } else {
            &TALLIES[index]
        };
        plan = append_fragment_phase(plan, op).expect("a tally phase");
        work.push(PhaseWork::Fragments(op));
    }
    let render: &'static RenderOp = &RENDERS[tallies - 1];
    plan = append_fragment_phase(plan, render).expect("a render phase");
    work.push(PhaseWork::Fragments(render));
    (plan, chain, work)
}

/// One renderer per plan length: it names the stream it reads and the phase
/// that wrote it, and both depend on how many tally phases came first.
static RENDERS: [RenderOp; 3] = [
    RenderOp("tally-a", 1),
    RenderOp("tally-b", 2),
    RenderOp("tally-c", 3),
];

/// Residency of every image at the moment each phase starts, so that "held one
/// phase too long" is a sampled fact rather than an end-state inference.
struct Residency {
    env: Arc<ArrayEnvironment>,
    at_phase: Mutex<Vec<(usize, Vec<bool>)>>,
    images: usize,
}

impl EventListener for Residency {
    fn on_event(&self, event: &Event) {
        if let Event::PhaseStarted { phase } = event {
            let live = (0..self.images)
                .map(|image| !self.env.is_discarded(image))
                .collect();
            self.at_phase.lock().unwrap().push((*phase, live));
        }
    }
}

fn residency_by_phase(reads: bool, tallies: usize) -> Vec<(usize, Vec<bool>)> {
    let (plan, chain, work) = tally_plan(reads, tallies);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env =
        Arc::new(ArrayEnvironment::for_decomposition(input().into(), &plan, [4, 4, 4]).unwrap());
    let watch = Arc::new(Residency {
        env: Arc::clone(&env),
        at_phase: Mutex::new(Vec::new()),
        images: plan.n_images(),
    });
    let listeners: Vec<Arc<dyn EventListener>> = vec![watch.clone()];
    execute_phases(
        "tally",
        &workflow,
        &plan,
        &Hints::default(),
        &*env,
        &listeners,
        &work,
    )
    .expect("a run");
    let seen = watch.at_phase.lock().unwrap().clone();
    seen
}

/// **The executor frees an image nothing reads as soon as it is written.**
///
/// This assertion used to encode the defect: `readers_of_image` credited phase
/// 1 with reading image 1, the executor held it through the whole of phase 1,
/// and the value on the last line was the opposite of what it is now. It is
/// **inverted rather than deleted**, so the residency it once recorded stays
/// readable as the thing that moved.
#[test]
fn a_phase_that_reads_no_pixels_is_not_credited_with_reading_its_image() {
    let (plan, _, _) = tally_plan(false, 1);
    assert!(
        !plan.phases[1].reads_input_image,
        "the plan records that phase 1 does no pixel IO"
    );
    assert_eq!(
        plan.readers_of_image(1),
        Vec::<usize>::new(),
        "and `readers_of_image` now agrees with it — this was `[1]`"
    );
    // image 1 dies after its writer, which is the zero-reader case of the same
    // rule. These three were `[0]`, `[1]` and `[2]`.
    assert_eq!(plan.images_dead_after(0), vec![0, 1]);
    assert_eq!(plan.images_dead_after(1), vec![2]);
    assert_eq!(plan.images_dead_after(2), vec![3]);

    // and the executor acts on it: image 1 is gone before phase 1 starts
    let held = residency_by_phase(false, 1);
    let phases: Vec<usize> = held.iter().map(|&(phase, _)| phase).collect();
    assert_eq!(phases, vec![0, 1, 2], "one sample per phase");
    assert!(held[0].1[1], "phase 0 is writing it");
    assert!(
        !held[1].1[1],
        "and it is dead by the time phase 1 starts — this was the assertion \
         that said it was still resident"
    );
}

/// The negative control: the same three phases with `reads_pixels() == true` in
/// the middle. Image 1 is resident during phase 1 here **because phase 1 reads
/// it**, so the residency above is not merely "this is how the executor always
/// behaves".
#[test]
fn the_same_plan_with_a_reading_phase_holds_the_image_for_a_reason() {
    let (plan, _, _) = tally_plan(true, 1);
    assert!(plan.phases[1].reads_input_image);
    assert_eq!(plan.readers_of_image(1), vec![1]);

    let held = residency_by_phase(true, 1);
    assert!(held[1].1[1], "phase 1 reads image 1");
    assert!(!held[2].1[1], "and it is dead once phase 1 has finished");

    // The two plans are now **distinguishable**, which is the fix stated as a
    // measurement. This was an `assert_eq!` while the executor could not tell
    // them apart: the whole defect was that a plan reading nothing and a plan
    // reading its image held the same bytes for the same time.
    assert_ne!(
        residency_by_phase(false, 1),
        held,
        "a phase that reads its image and a phase that does not must not have \
         the same residency"
    );
}

/// **There is no over-hold left, however many phases read nothing.**
///
/// Five phases, three of which are fragment phases doing no pixel IO at all.
/// Every one of them is now credited with reading nothing, and image 1 — the
/// only image anything allocates after the input — dies with the phase that
/// wrote it. Images 2, 3 and 4 are slots those phases never fill, since a
/// fragment phase that writes no pixels writes no image.
///
/// **Inverted, not deleted.** This test measured the over-hold: each of those
/// phases was credited with `vec![phase]`, image 1's `first_free` was `2`
/// rather than `1`, and the arithmetic on the line below came out `1` phase
/// rather than `0`. One phase was the ceiling as well as the measurement,
/// because image `p`'s only credited reader was phase `p` and its writer is
/// phase `p - 1` — the claim could be wrong, but not wrong by two. The bytes
/// that one phase cost are kept at the bottom, because they are what made the
/// fix worth making and they do not stop being true for having been fixed.
#[test]
fn no_image_outlives_the_phase_that_wrote_it_when_nothing_reads_it() {
    let (plan, _, _) = tally_plan(false, 3);
    assert_eq!(plan.n_phases(), 5);
    assert_eq!(plan.n_images(), 6);

    for phase in 1..plan.n_phases() {
        assert!(
            !plan.phases[phase].reads_input_image,
            "phase {phase} does no pixel IO"
        );
        assert_eq!(
            plan.readers_of_image(phase),
            Vec::<usize>::new(),
            "and is credited with reading nothing — this was `[{phase}]`"
        );
    }

    let held = residency_by_phase(false, 3);
    assert_eq!(held.len(), 5, "one sample per phase");

    // Image 1 is written by phase 0 and is gone by the time phase 1 starts.
    let writer = 0;
    let first_free = held
        .iter()
        .find(|(_, live)| !live[1])
        .map(|&(phase, _)| phase)
        .expect("image 1 is freed during the run");
    assert_eq!(first_free, 1, "this was 2");
    assert_eq!(
        first_free - writer - 1,
        0,
        "no phase longer than it needs; this was 1"
    );

    // What that phase of residency cost, which is why the fix was worth
    // making: one whole image at the plan's element type. `forme.md`'s
    // residency table calls `1024^3` the largest volume a large node affords,
    // which is the figure to read the second line as.
    let bytes = |volume: [usize; 3]| {
        volume.iter().product::<usize>() as u64 * std::mem::size_of::<f64>() as u64
    };
    assert_eq!(bytes(VOLUME), 15_360);
    assert_eq!(bytes([1024, 1024, 1024]), 8 << 30, "8 GiB");
}

/// **This was the specification of the fix, and the fix has landed.** Kept
/// live and kept whole, because the argument is what makes the three
/// assertions above reviewable rather than arbitrary.
///
/// The defect was in [`Decomposition::readers_of_image`]
/// (`src/decomposition.rs`): it counted phase `p` as a reader of image `p`
/// unconditionally, and a fragment phase declaring `reads_pixels() == false`
/// reads no pixels at all. `reads_input_image` on the phase already recorded
/// the truth, and `fragment_phase` is what put it there.
///
/// **Three things had to move together**, which is why this was not done
/// inline, and all three did.
/// Correcting `readers_of_image` alone leaves image 1 with *no* reader, and
/// `images_dead_after` answered `readers.last() == Some(&phase)` — so an image
/// with no reader would be named by no phase and the executor would never free
/// it, which is worse than the over-count. The zero-reader rule
/// `peak_image_bytes` already applies — *an image nothing reads dies as soon as
/// it is written* — had to move into `images_dead_after` in the same change,
/// and did.
///
/// **And the `phase != image` guard had to go with them.** A phase that reads
/// an image through `source_images` rather than as its input image was skipped
/// by that guard and was covered only by the unconditional push; narrowing the
/// push without widening the loop drops a real reader. This was measured, not
/// reasoned: it failed every run in `tests/region_tabulation.rs`, each with
/// `image 1 was discarded after phase 0`. The predicate that survives all three
/// is *phase `p` reads image `i` when `i == p` and `reads_input_image`, or when
/// `source_images` names `i`* — one loop, no special case, and it is what
/// `readers_of_image` now is.
///
/// # The blast radius, predicted from a patched copy and then observed
///
/// Measured before the change by applying it in an isolated copy of the crate
/// and running the whole suite: exactly two files move and no other, and no
/// output voxel anywhere moves. When the change landed, exactly those
/// assertions moved and no others — the prediction is left here in full
/// because a blast radius that turned out right is the evidence that the
/// method was sound.
///
/// In `tests/source_leaf.rs`, two assertions, both because the **output** image
/// has no reader inside the run and is now named by the phase that wrote it —
/// harmlessly, since `image_visibility` is what keeps it alive and both
/// `strategy.rs` and `peak_image_bytes` already consult it:
///
/// * `images_dead_after(2)`: `[1, 2]` becomes `[1, 2, 3]`
/// * the plain-chain loop's `images_dead_after(phase) == [phase]`: at the last
///   phase, `[2]` becomes `[2, 3]`
///
/// In this file, the three tests above, which is what they were for. Each has
/// since been inverted in place, so the old value is on the left:
///
/// * `readers_of_image(1)`: `[1]` becomes `[]`
/// * `images_dead_after(0)`: `[0]` becomes `[0, 1]`; `(1)`: `[1]` becomes
///   `[2]`; `(2)`: `[2]` becomes `[3]`
/// * the sampled residency of image 1: still live at the start of phase 1
///   becomes already freed, so `first_free` is `1` rather than `2` and the
///   over-hold is `0` phases rather than `1`
/// * the two runs of `the_same_plan_with_a_reading_phase_holds_the_image_for_a_reason`
///   stop being equal, which is that test's whole point
///
/// **Nothing in the seven tests this file opened with moves**, and the reason
/// is structural rather than lucky: every plan in them is pixel phases only, so
/// `reads_input_image` is true everywhere and the corrected predicate returns
/// exactly what the unconditional push returned.
#[test]
fn a_phase_that_reads_no_pixels_is_not_a_reader_of_its_input_image() {
    let (plan, _, _) = tally_plan(false, 1);
    assert_eq!(
        plan.readers_of_image(1),
        Vec::<usize>::new(),
        "phase 1 does no pixel IO, so nothing reads image 1"
    );
    // dead after its writer, which is the zero-reader case of the same rule
    assert_eq!(plan.images_dead_after(0), vec![0, 1]);
    // image 2 is the slot phase 1 never fills — a fragment phase that writes no
    // pixels writes no image. It has no reader either, so the same rule names it
    // here, and naming a slot nothing allocated costs nothing.
    assert_eq!(plan.images_dead_after(1), vec![2]);

    // and the executor frees it a phase earlier than it used to
    let held = residency_by_phase(false, 1);
    assert!(
        !held[1].1[1],
        "image 1 is dead before phase 1 starts, because phase 1 never reads it"
    );

    // the negative control does not move: a phase that does read it still holds it
    let control = residency_by_phase(true, 1);
    assert!(control[1].1[1], "phase 1 reads image 1 here");
    assert_ne!(
        held, control,
        "and the two plans are distinguishable, which is the point of the fix"
    );
}

/// **A caller can release the input, and the bytes come back.**
///
/// [`Visibility`] is derived from *position* — image 0 is the run's input, so it
/// is `Published` and the executor never frees it. That is the right default and
/// it is not a statement about whether anybody wants it. `Hints::release_images`
/// is how a caller says it does not, and it is the only way the input's bytes
/// are ever returned during a run: at one block an image is a whole volume, and
/// image 0 is then the largest single thing the executor holds.
///
/// Three assertions and the last two are what make the first mean something.
///
/// * The bytes fall. `allocated_image_bytes` sums only images that hold a
///   buffer, so it is a measurement of occupancy rather than of a flag —
///   `resident_images` counts non-discarded images and would say the same thing
///   whether or not anything was freed.
/// * **The control.** The identical run without the hint holds image 0 to the
///   end. Without this the first assertion would pass just as well against an
///   environment that never allocated image 0 in the first place.
/// * **Not one voxel moves.** Releasing an image after its last reader cannot
///   change an answer, and this is where that is checked rather than argued.
#[test]
fn releasing_the_input_frees_it_after_its_last_reader_and_moves_no_voxel() {
    let released = Hints {
        release_images: [ImageId::from(0usize)].into_iter().collect(),
        ..Hints::default()
    };
    let (with_answer, with) = run(&released);
    let (without_answer, without) = run(&Hints::default());

    let image_bytes = (VOLUME.iter().product::<usize>() * Dtype::F64.size_of()) as u64;

    // The control first, so that a fall measured below is a fall from somewhere.
    assert!(
        !without.is_discarded(0),
        "the default must still hold image 0"
    );
    assert_eq!(
        without.allocated_image_bytes(),
        2 * image_bytes,
        "the default ends holding the input and the output"
    );

    assert!(with.is_discarded(0), "the hint did not reach the executor");
    assert_eq!(
        with.allocated_image_bytes(),
        image_bytes,
        "releasing the input must give back exactly one whole image's bytes, and \
         `allocated_image_bytes` counts buffers rather than flags"
    );
    assert_eq!(
        without.allocated_image_bytes() - with.allocated_image_bytes(),
        image_bytes
    );

    // And the answer is the answer.
    assert_eq!(with_answer, without_answer);
    assert_eq!(with_answer, input());

    // Reading it back is loud, for `reading_a_freed_image_fails_and_says_why`'s
    // reason: a released image that came back as zeros would be the defect this
    // crate fills unwritten images with NaN to prevent.
    let region = blockflow::region::Region::new(&[0, 0, 0], &[4, 4, 4]);
    let message = with.read(0, &region).unwrap_err().to_string();
    assert!(message.contains("discarded"), "{message}");
}

/// **`keep_images` wins over `release_images`.**
///
/// A caller that names one image in both has contradicted itself, and the
/// reading that cannot lose data is the one to take. Asserted rather than left
/// to the reader of the guard, because it is the kind of precedence that gets
/// inverted by a refactor and noticed by nobody.
#[test]
fn keeping_an_image_beats_releasing_it() {
    let both = Hints {
        keep_images: [ImageId::from(0usize)].into_iter().collect(),
        release_images: [ImageId::from(0usize)].into_iter().collect(),
        ..Hints::default()
    };
    let (_, env) = run(&both);
    assert!(
        !env.is_discarded(0),
        "`release_images` overrode `keep_images`, so a caller that asked to keep an image lost it"
    );

    // The liveness half: releasing alone does free it, so the assertion above is
    // about the precedence and not about a hint that does nothing.
    let released = Hints {
        release_images: [ImageId::from(0usize)].into_iter().collect(),
        ..Hints::default()
    };
    let (_, alone) = run(&released);
    assert!(alone.is_discarded(0));
}
