// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A run seeded with more than one array.**
//
// Before this, a run had exactly one input. `ArrayEnvironment::new` and
// `ArrayEnvironment::for_decomposition` took a single `Voxels` as image 0 and
// created every other image `pending`, to be written by a phase;
// `ZarrEnvironment::create` did the same on disk. `BlockOp::source_inputs` and
// `Chain::Source` read *additional images of the same plan* — by definition
// something a phase wrote — so there was no way to hand the framework a second
// array that already existed.
//
// What this file pins, in the order the claims depend on each other:
//
// 1. **An N-way Boolean union runs as a blockable phase.** Every part of it was
//    already here: `LogicCombine` accepts `inputs.len() >= 2` and folds left,
//    `Chain::parallel` takes N branches, and a `Chain::Source` leaf is legal as
//    a branch. The only missing piece was something for the leaves to point at.
// 2. **It is decomposition-invariant and byte-identical to a resident
//    reference** — the same fold computed in one call over whole arrays.
// 3. **Each leaf reads the array it names.** `OR` is idempotent, so a union
//    that quietly handed every leaf the same buffer would agree with the
//    reference wherever the arrays overlap; the negative control replaces one
//    input and requires the answer to move.
// 4. **A supplied input's element type is its own.** Image 0 here holds `f64`
//    and every supplied array holds `bool`, so an implementation that assumed
//    the input's type — or the plan's fold — cannot pass.
// 5. **A supplied input is never freed and never written**, which is the whole
//    of what `ImageKind::Input` says that `Visibility::Published` cannot: an
//    intermediate may be dropped and recomputed, and an input may not be
//    recomputed at any price.
// 6. **The refusals**: a wrong count, a wrong shape, a wrong element type, and
//    a reader that does not say what the array holds — each by name, at plan or
//    prepare time, before a block runs.
//
// No assertion here is on wall-clock time.

use ndarray::Array3;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::decomposition::{Decomposition, ImageKind, PhaseDecomposition, Visibility};
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::{Logic, LogicCombine, NarrowOp, VoxelwiseMapOp};
use blockflow::probes::IdentityOp;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [16, 12, 10];

/// Five arrays in the union: one the run is seeded with and four handed to it.
///
/// Five rather than two because `LogicCombine` folds left over `n` branches and
/// a two-branch fold is the diamond that already worked; the point of the gap
/// was arity.
const CHANNELS: usize = 5;

// ------------------------------------------------------------- fixtures --

/// Channel `which` as an intensity volume: separated boxes on a lattice whose
/// period differs per channel.
///
/// **Different periods, deliberately.** Channels that agreed everywhere would
/// make the union equal to any one of them, and every comparison below would be
/// true for a reason that has nothing to do with the fold. These overlap
/// partially, so the union is a strict superset of each and of no two.
fn channel(which: usize) -> Array3<f64> {
    let period = 3 + which;
    let phase = which * 2;
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(i, j, k)| {
        let hit = (i + phase) % period == 0 && (j + which) % (period + 1) < 2 && k % 4 < 3;
        f64::from(hit) * (1.0 + which as f64)
    })
}

/// The same channel as the mask a threshold would make of it.
fn channel_mask(which: usize) -> Array3<bool> {
    channel(which).mapv(|value| value > 0.5)
}

/// A supplied array: channel `which`, already binary, as a `bool` image.
fn supplied(which: usize) -> Voxels {
    channel_mask(which).into()
}

/// Slot 0: threshold channel 0 to `0.0`/`1.0`. Still `f64`.
fn threshold() -> Chain {
    Chain::op(VoxelwiseMapOp::new("threshold", |value| {
        f64::from(value > 0.5)
    }))
}

/// Slot 1: narrow the thresholded channel to `bool`, so that the image the
/// union phase is handed holds what the supplied arrays hold.
fn to_mask() -> Chain {
    Chain::op(NarrowOp::to_mask("to_mask"))
}

/// Slot 2: the union itself — the phase's own input beside `CHANNELS - 1`
/// supplied arrays, folded by `OR`.
///
/// The first branch is an identity over the image the phase is handed rather
/// than a fourth source leaf, so that the phase reads its input the way every
/// other phase does and the fan-in is genuinely one computed arm plus N stored
/// ones.
fn union(images: &[usize]) -> Chain {
    let mut branches = vec![Chain::op(IdentityOp::new("kept", [0, 0, 0]))];
    for &image in images {
        branches.push(Chain::source(image, Dtype::Bool));
    }
    Chain::parallel(branches, Box::new(LogicCombine::new("or", Logic::Or))).expect("a fan-in")
}

/// The addresses of the supplied arrays, in the order they are handed over.
fn supplied_images() -> Vec<usize> {
    (0..CHANNELS - 1)
        .map(|which| ImageId::supplied(which).index())
        .collect()
}

fn chain() -> Chain {
    Chain::sequence(vec![threshold(), to_mask(), union(&supplied_images())])
}

/// One phase per slot, so the union really reads a materialised image and the
/// supplied arrays are images rather than buffers inside a phase.
fn one_phase_per_slot(chain: &Chain, grid: &BlockGrid) -> Decomposition {
    let slots = chain.slots();
    let phases = (0..slots.len())
        .map(|slot| {
            PhaseDecomposition::derive(
                vec![slot],
                vec![slots[slot].display_name()],
                [0usize, 0, 0],
                [0usize, 0, 0],
                grid.clone(),
            )
        })
        .collect();
    let mut plan = Decomposition {
        volume: VOLUME,
        dtype: Dtype::F64,
        phases,
        chain_reach: [0, 0, 0],
    };
    plan.declare_dtypes(chain).expect("element types");
    plan.declare_source_images(chain).expect("source images");
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

fn arrays() -> Vec<Voxels> {
    (1..CHANNELS).map(supplied).collect()
}

fn run_with(
    chain: Chain,
    grid: &BlockGrid,
    inputs: Vec<Voxels>,
    hints: &Hints,
) -> (Array3<bool>, ArrayEnvironment) {
    let plan = one_phase_per_slot(&chain, grid);
    let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
    let env =
        ArrayEnvironment::with_inputs(channel(0).into(), inputs, &plan, [4, 4, 4]).expect("an env");
    execute("union", &workflow, &plan, hints, &env).expect("a run");
    let out = env.output().view::<bool>().unwrap().to_owned();
    (out, env)
}

fn run(grid: &BlockGrid) -> (Array3<bool>, ArrayEnvironment) {
    run_with(chain(), grid, arrays(), &Hints::default())
}

/// The union computed in one call over whole arrays, with no plan and no
/// blocking anywhere in it.
///
/// **A resident reference, not another decomposition.** Two decompositions
/// agreeing proves they agree; this is the arithmetic the fold is supposed to
/// be, written out.
fn resident_union() -> Array3<bool> {
    let mut out = channel_mask(0);
    for which in 1..CHANNELS {
        let other = channel_mask(which);
        for (value, &bit) in out.iter_mut().zip(other.iter()) {
            *value |= bit;
        }
    }
    out
}

// ------------------------------------- 1 and 2. the union, and invariance --

/// The acceptance test for the gap: an N-channel Boolean union running as a
/// blockable phase, byte-identical to the resident fold, under every cut.
#[test]
fn an_n_channel_union_is_the_resident_answer_under_every_decomposition() {
    let wanted = resident_union();
    // Not a constant and not one of its own operands, or every comparison here
    // would be true for the wrong reason.
    assert!(wanted.iter().any(|&bit| bit));
    assert!(wanted.iter().any(|&bit| !bit));
    for which in 0..CHANNELS {
        assert_ne!(wanted, channel_mask(which), "the union is channel {which}");
    }

    for grid in grids() {
        let (out, _) = run(&grid);
        assert_eq!(out, wanted, "block {:?}", grid.block());
    }
}

/// Every phase of the run really is blocked: the whole-volume cut and a
/// four-voxel cut do not run the same number of tasks.
#[test]
fn the_union_phase_is_cut_into_blocks_like_any_other() {
    let whole = one_phase_per_slot(&chain(), &BlockGrid::new(VOLUME, VOLUME).unwrap());
    let cut = one_phase_per_slot(&chain(), &BlockGrid::along(VOLUME, &[0, 1, 2], 4).unwrap());
    assert_eq!(whole.n_tasks(), whole.n_phases());
    assert!(cut.n_tasks() > 4 * whole.n_tasks());
    // and the union phase names every supplied array
    assert_eq!(cut.phases[2].source_images, supplied_images());
    assert_eq!(cut.n_supplied_inputs(), CHANNELS - 1);
    assert_eq!(cut.supplied_input_images(), supplied_images());
}

// ----------------------------------------------- 3. the negative control --

/// Each leaf reads the array it names.
///
/// `OR` is idempotent and monotone, so a union handed the same buffer for every
/// leaf still agrees with the reference wherever the channels overlap. Swapping
/// one supplied array for a different one has to move the answer, and swapping
/// two of them for each other must not — the fold is commutative, and a
/// difference there would mean the leaves were being paired with the wrong
/// arrays by position rather than by address.
#[test]
fn each_leaf_reads_the_array_it_names() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let (reference, _) = run(&grid);

    // One array replaced by an empty one: strictly less is set.
    let mut fewer = arrays();
    fewer[CHANNELS - 2] = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2]), false).into();
    let (dropped, _) = run_with(chain(), &grid, fewer, &Hints::default());
    assert_ne!(dropped, reference);
    assert!(dropped
        .iter()
        .zip(reference.iter())
        .all(|(&less, &more)| !less || more));

    // Two of them exchanged: the fold is commutative, so the answer must not
    // move. If it did, the leaves would be reading by position.
    let mut swapped = arrays();
    swapped.swap(0, CHANNELS - 2);
    let (exchanged, _) = run_with(chain(), &grid, swapped, &Hints::default());
    assert_eq!(exchanged, reference);
}

/// The union is not the phase's own input read `CHANNELS` times.
#[test]
fn a_union_of_supplied_arrays_is_not_a_union_of_the_input_with_itself() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let (out, _) = run(&grid);
    assert_ne!(out, channel_mask(0));
    assert!(out
        .iter()
        .zip(channel_mask(0).iter())
        .any(|(&union, &own)| union && !own));
}

// --------------------------------------- 4. a supplied type is its own --

/// Image 0 holds `f64` and every supplied array holds `bool`.
///
/// The plan folds the chain to answer for the images it writes, and there is
/// nothing to fold for an array no phase wrote — so the readers' declaration is
/// what `dtype_at` answers with, and this is the assertion that it is not image
/// 0's type wearing a different name.
#[test]
fn a_supplied_input_holds_its_own_element_type() {
    let plan = one_phase_per_slot(&chain(), &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    assert_eq!(plan.dtype_at(0), Dtype::F64);
    assert_eq!(plan.dtype_at(1), Dtype::F64);
    assert_eq!(plan.dtype_at(2), Dtype::Bool);
    for &image in &supplied_images() {
        assert_eq!(plan.dtype_at(image), Dtype::Bool);
        assert_eq!(plan.volume_at(image), VOLUME);
    }
    assert_eq!(
        plan.phases[2].supplied_dtypes,
        supplied_images()
            .into_iter()
            .map(|image| (image, Dtype::Bool))
            .collect::<Vec<_>>()
    );
}

// ------------------------------------------- 5. an input is not the run's --

/// The three-way kind, and the one thing it says that `Visibility` cannot.
#[test]
fn a_supplied_input_is_an_input_and_the_output_is_an_output() {
    let plan = one_phase_per_slot(&chain(), &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    assert_eq!(plan.image_kind(0), ImageKind::Input);
    assert_eq!(plan.image_kind(1), ImageKind::Intermediate);
    assert_eq!(plan.image_kind(2), ImageKind::Intermediate);
    assert_eq!(plan.image_kind(3), ImageKind::Output);
    for &image in &supplied_images() {
        assert_eq!(plan.image_kind(image), ImageKind::Input);
        // and the coarser question still answers the way everything that reads
        // it expects: nothing outside this file moves.
        assert_eq!(plan.image_visibility(image), Visibility::Published);
    }
    assert_eq!(plan.image_visibility(1), Visibility::Internal);
    assert_eq!(plan.image_visibility(3), Visibility::Published);
}

/// A supplied input survives a run that frees every intermediate it can, and
/// refuses to be freed if something asks.
#[test]
fn a_supplied_input_is_never_freed_and_never_written() {
    let grid = BlockGrid::along(VOLUME, &[0], 4).unwrap();
    let (_, env) = run(&grid);
    for &image in &supplied_images() {
        assert!(!env.is_discarded(image));
        assert_eq!(env.image(image).shape(), VOLUME);
    }
    // an intermediate did go, which is what makes the line above a claim
    assert!(env.is_discarded(1));

    let image = supplied_images()[0];
    let refusal = env
        .discard_image(image)
        .expect_err("an input cannot be freed");
    let message = refusal.to_string();
    assert!(message.contains("supplied input 0"), "{message}");
    assert!(message.contains("no phase writes it"), "{message}");
}

// -------------------------------------------------------- 6. refusals --

fn refusal_of(inputs: Vec<Voxels>) -> String {
    let plan = one_phase_per_slot(&chain(), &BlockGrid::along(VOLUME, &[0], 4).unwrap());
    match ArrayEnvironment::with_inputs(channel(0).into(), inputs, &plan, [4, 4, 4]) {
        Ok(_) => panic!("a refusal"),
        Err(refusal) => refusal.to_string(),
    }
}

#[test]
fn too_few_supplied_arrays_is_refused_by_name() {
    let mut short = arrays();
    short.pop();
    let message = refusal_of(short);
    assert!(message.contains("handed 3 array(s)"), "{message}");
    assert!(message.contains("supplied input 3"), "{message}");
}

#[test]
fn a_supplied_array_of_the_wrong_shape_is_refused_by_name() {
    let mut wrong = arrays();
    wrong[1] = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2] - 1), false).into();
    let message = refusal_of(wrong);
    assert!(message.contains("supplied input 1"), "{message}");
    assert!(message.contains("coordinate space"), "{message}");
}

#[test]
fn a_supplied_array_of_the_wrong_element_type_is_refused_by_name() {
    let mut wrong = arrays();
    wrong[2] = channel(3).into();
    let message = refusal_of(wrong);
    assert!(message.contains("supplied input 2"), "{message}");
    assert!(message.contains("float64"), "{message}");
    assert!(message.contains("bool"), "{message}");
}

/// A reader that names a supplied array without saying what is in it.
///
/// There is no fold to ask — no phase wrote it — so the declaration is the only
/// statement there is, and a plan without one would have `dtype_at` guessing.
#[test]
fn a_supplied_input_nobody_declares_the_type_of_is_refused_by_name() {
    use blockflow::error::Result;
    use blockflow::op::{Anchor, BlockOp, SourceInput, SourceInputs};

    struct Silent(usize);
    impl BlockOp for Silent {
        fn name(&self) -> &'static str {
            "silent"
        }
        fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
            0
        }
        fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
            vec![SourceInput::voxelwise(self.0)]
        }
        fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
            out.assign(input)
        }
        fn apply_with(
            &self,
            input: &Voxels,
            _sources: SourceInputs<'_>,
            out: &mut Voxels,
            _at: &Anchor,
        ) -> Result<()> {
            out.assign(input)
        }
        fn cost_per_voxel(&self) -> f64 {
            1.0
        }
    }

    let mut plan = PlanBuilder::new(
        VOLUME,
        Dtype::F64,
        BlockGrid::along(VOLUME, &[0], 4).unwrap(),
    );
    plan.pixels(Chain::op(Silent(ImageId::supplied(0).index())))
        .expect("a phase");
    let message = match plan.finish() {
        Ok(_) => panic!("a refusal"),
        Err(refusal) => refusal.to_string(),
    };
    assert!(message.contains("supplied input 0"), "{message}");
    assert!(message.contains("nothing says what it holds"), "{message}");
}

/// Two readers of one supplied array cannot disagree about what is in it.
#[test]
fn two_readers_declaring_different_types_are_refused_by_name() {
    let image = ImageId::supplied(0).index();
    let chain = Chain::parallel(
        vec![
            Chain::source(image, Dtype::Bool),
            Chain::source(image, Dtype::F64),
        ],
        Box::new(LogicCombine::new("or", Logic::Or)),
    )
    .expect("a fan-in");
    let message = chain
        .source_inputs(VOLUME)
        .expect_err("a refusal")
        .to_string();
    assert!(message.contains("supplied input 0"), "{message}");
    assert!(
        message.contains("only one of them can be right"),
        "{message}"
    );
}

/// The plan builder hands out supplied addresses before a phase exists, which
/// is what makes them usable: the ops that read one are built first.
#[test]
fn the_builder_assembles_a_plan_that_reads_a_supplied_array() {
    let mut plan = PlanBuilder::new(
        VOLUME,
        Dtype::F64,
        BlockGrid::along(VOLUME, &[0], 4).unwrap(),
    );
    plan.pixels(threshold()).expect("a phase");
    plan.pixels(to_mask()).expect("a phase");
    plan.pixels(union(&supplied_images())).expect("a phase");
    let assembly = plan.finish().expect("a plan");
    assert_eq!(assembly.decomposition.n_supplied_inputs(), CHANNELS - 1);

    let env = ArrayEnvironment::with_inputs(
        channel(0).into(),
        arrays(),
        &assembly.decomposition,
        [4, 4, 4],
    )
    .expect("an env");
    execute(
        "built",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
    )
    .expect("a run");
    assert_eq!(
        env.output().view::<bool>().unwrap().to_owned(),
        resident_union()
    );
}

// ------------------------------------------------------------ on storage --

/// The same claim against the storage backend, because the gap was stated about
/// both constructors and closing it in memory only would close half of it.
#[cfg(feature = "zarr")]
mod on_disk {
    use super::*;
    use blockflow::zarr_env::ZarrEnvironment;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A directory nobody else is using, removed even if a test panics.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let unique = NEXT.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "blockflow-supplied-{}-{name}-{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_stored_run_seeded_with_several_arrays_gives_the_resident_answer() {
        let wanted = resident_union();
        for grid in grids() {
            let scratch = Scratch::new("union");
            let chain = chain();
            let plan = one_phase_per_slot(&chain, &grid);
            let workflow = Workflow::new(chain, VOLUME, Dtype::F64);
            let input: Voxels = channel(0).into();
            let held = arrays();
            let supplied: Vec<&Voxels> = held.iter().collect();
            let env = ZarrEnvironment::create_with_inputs(&scratch.0, &input, &supplied, [4, 4, 4])
                .expect("a store");
            execute("union", &workflow, &plan, &Hints::default(), &env).expect("a run");
            let out = env
                .image(plan.n_images() - 1)
                .expect("the output image")
                .view::<bool>()
                .unwrap()
                .to_owned();
            assert_eq!(out, wanted, "block {:?}", grid.block());
        }
    }

    #[test]
    fn a_stored_supplied_input_is_neither_written_nor_freed() {
        let scratch = Scratch::new("refusals");
        let chain = chain();
        let plan = one_phase_per_slot(&chain, &BlockGrid::along(VOLUME, &[0], 4).unwrap());
        let input: Voxels = channel(0).into();
        let held = arrays();
        let supplied: Vec<&Voxels> = held.iter().collect();
        let env = ZarrEnvironment::create_with_inputs(&scratch.0, &input, &supplied, [4, 4, 4])
            .expect("a store");
        env.prepare(&plan).expect("a plan this store can host");

        let image = supplied_images()[0];
        let message = match env.discard_image(image) {
            Ok(()) => panic!("an input cannot be freed"),
            Err(refusal) => refusal.to_string(),
        };
        assert!(message.contains("supplied input 0"), "{message}");
        assert!(message.contains("no phase writes it"), "{message}");

        // and it is still readable afterwards, which is what "refused" has to
        // mean rather than "refused after erasing it"
        assert_eq!(env.image_shape(image).expect("a shape"), VOLUME);
    }

    #[test]
    fn a_stored_array_of_the_wrong_shape_is_refused_by_name() {
        let scratch = Scratch::new("shape");
        let input: Voxels = channel(0).into();
        let short: Voxels = Array3::from_elem((VOLUME[0], VOLUME[1], VOLUME[2] - 1), false).into();
        let held = arrays();
        let mut supplied: Vec<&Voxels> = held.iter().collect();
        supplied[2] = &short;
        let message =
            match ZarrEnvironment::create_with_inputs(&scratch.0, &input, &supplied, [4, 4, 4]) {
                Ok(_) => panic!("a refusal"),
                Err(refusal) => refusal.to_string(),
            };
        assert!(message.contains("supplied input 2"), "{message}");
        assert!(message.contains("coordinate space"), "{message}");
    }
}

// --------------------------------------------- a fragment phase reads one --

/// A fragment op can name a supplied array too, and it declares what is in it.
///
/// The chain half and the fragment half record `source_images` in two different
/// places for a stated reason — a fragment op declares its second image on
/// itself and only the `(plan, work)` pair holds the op — so both halves have to
/// know that an image nothing wrote is an input rather than a forward reference.
mod fragment_phase {
    use super::*;
    use blockflow::error::Result;
    use blockflow::fragment::{
        BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, SeamFold, SourceBlocks,
    };
    use blockflow::op::SourceInput;
    use blockflow::sidecar::Lifecycle;

    /// Counts the set positions of a supplied array, per block.
    struct CountOp {
        image: usize,
        /// `false` leaves the element type unsaid, which is the thing that is
        /// refused: nothing folds one for an array no phase wrote.
        says_what_it_holds: bool,
    }

    impl FragmentOp for CountOp {
        fn name(&self) -> &'static str {
            "count"
        }

        fn reads_pixels(&self) -> bool {
            false
        }

        fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
            let declared = SourceInput::voxelwise(self.image);
            vec![match self.says_what_it_holds {
                true => declared.holding(Dtype::Bool),
                false => declared,
            }]
        }

        fn seam_fold(&self) -> Option<SeamFold> {
            Some(SeamFold::PerBlock)
        }

        fn outputs(&self) -> Vec<FragmentOutput> {
            vec![FragmentOutput::new(
                "counts",
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            )]
        }

        fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
            unreachable!("this op declares an operand and runs through `apply_with`")
        }

        fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
            let array = sources.get(self.image)?;
            let set = array
                .as_array()?
                .view::<bool>()?
                .iter()
                .filter(|&&bit| bit)
                .count();
            let _ = at.index;
            Ok(BlockOutput::fragment(
                "counts",
                (set as u64).to_le_bytes().to_vec(),
            ))
        }
    }

    #[test]
    fn a_fragment_phase_may_read_a_supplied_array_and_records_what_it_holds() {
        let image = ImageId::supplied(0).index();
        let mut plan = PlanBuilder::new(
            VOLUME,
            Dtype::F64,
            BlockGrid::along(VOLUME, &[0], 4).unwrap(),
        );
        plan.pixels(threshold()).expect("a phase");
        plan.fragments(CountOp {
            image,
            says_what_it_holds: true,
        })
        .expect("a fragment phase");
        let assembly = plan.finish().expect("a plan");
        let phase = &assembly.decomposition.phases[1];
        assert_eq!(phase.source_images, vec![image]);
        assert_eq!(phase.supplied_dtypes, vec![(image, Dtype::Bool)]);
        assert_eq!(assembly.decomposition.dtype_at(image), Dtype::Bool);
        assert_eq!(assembly.decomposition.n_supplied_inputs(), 1);
    }

    #[test]
    fn a_fragment_op_that_does_not_say_what_it_holds_is_refused_by_name() {
        let mut plan = PlanBuilder::new(
            VOLUME,
            Dtype::F64,
            BlockGrid::along(VOLUME, &[0], 4).unwrap(),
        );
        plan.pixels(threshold()).expect("a phase");
        let message = match plan.fragments(CountOp {
            image: ImageId::supplied(0).index(),
            says_what_it_holds: false,
        }) {
            Ok(_) => panic!("a refusal"),
            Err(refusal) => refusal.to_string(),
        };
        assert!(message.contains("supplied input 0"), "{message}");
        assert!(message.contains("nothing says what it holds"), "{message}");
    }
}
