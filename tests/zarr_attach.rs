// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **Binding a run to arrays that already exist.**
//
// Every other `ZarrEnvironment` constructor takes `input: &Voxels` — the whole
// of image 0, in memory — so the one case this crate could not start from was
// data already too large to hold, which is the case it exists for. A slide at
// `66048 x 157440` in four channels is 41 GB before anything is computed.
//
// `ZarrEnvironment::attach` closes that, and its acceptance criterion is the
// one `zarr_env` already states for itself and is asserted first below: **where
// the bytes came from is invisible to the answer.** A chain over an array
// written by `create` and re-opened by `attach` produces what the same chain
// produces without the round trip, voxel for voxel.
//
// The rest is the window — the sub-box of a stored array that one image is —
// which is one feature answering three needs: one channel of an OME-Zarr
// `[c, y, x]` level, a region of a large image to iterate on, and a pyramid
// level. Each is tested for the thing that could be silently wrong, which in
// every case is an offset applied in one place and forgotten in another.

#![cfg(feature = "zarr")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array3;

use blockflow::decomposition::{Decomposition, PhaseDecomposition};
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::{ElementShape, StructuringElement};
use blockflow::ops::rank::RankFilterOp;
use blockflow::strategy::{execute, Hints, Workflow};
use blockflow::synthetic::{Scene, SceneSpec};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Dtype, Region};

const VOLUME: [usize; 3] = [16, 20, 24];

// ------------------------------------------------------------- fixtures --

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "blockflow-zarr-attach-{}-{name}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    fn path(&self) -> &PathBuf {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn intensities(shape: [usize; 3]) -> Voxels {
    let scene = Scene::new(
        SceneSpec::new(shape, 20260821)
            .with_objects(25)
            .with_radius(1.5, 3.5)
            .with_noise(0.02),
    )
    .unwrap();
    let rendered = scene.render();
    let mut array = Array3::zeros((shape[0], shape[1], shape[2]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                array[[i, j, k]] = rendered.intensity[[i, j, k]];
            }
        }
    }
    array.into()
}

fn median_workflow(volume: [usize; 3]) -> Workflow {
    let element = StructuringElement::from_radius(ElementShape::Box, [1, 1, 1]);
    Workflow::new(
        Chain::op(RankFilterOp::median("median", element)),
        volume,
        Dtype::F64,
    )
}

fn plan(workflow: &Workflow, volume: [usize; 3], block: usize) -> Decomposition {
    let reach = workflow.chain.reach3(&volume);
    let slots = workflow.chain.slots();
    let names: Vec<String> = slots.iter().map(|slot| slot.display_name()).collect();
    let grid = BlockGrid::along(volume, &[0, 1, 2], block).unwrap();
    let phase = PhaseDecomposition::derive((0..slots.len()).collect(), names, reach, reach, grid);
    Decomposition {
        volume,
        dtype: workflow.dtype,
        phases: vec![phase],
        chain_reach: reach,
    }
}

/// The message of a refusal.
///
/// `unwrap_err` would need `ZarrEnvironment: Debug`, which it is not and has no
/// reason to be — it holds open arrays and a lock table.
fn refusal(outcome: blockflow::Result<ZarrEnvironment>) -> String {
    match outcome {
        Ok(_) => panic!("expected a refusal and got an environment"),
        Err(error) => error.to_string(),
    }
}

fn assert_same(left: &Voxels, right: &Voxels, what: &str) {
    assert_eq!(left.dtype(), right.dtype(), "{what}: element type");
    assert_eq!(left.shape(), right.shape(), "{what}: shape");
    let left_values = left.widened();
    let right_values = right.widened();
    for (index, (a, b)) in left_values.iter().zip(right_values.iter()).enumerate() {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "{what}: element {index} is {a} against {b}"
        );
    }
}

/// Write `input` as an array on a disk and return the directory holding it.
///
/// `create` is used rather than `zarrs` directly so that what is attached to is
/// an array this crate wrote, which is what makes the comparison below a
/// statement about `attach` alone.
fn stored(root: &PathBuf, input: &Voxels, chunk: [usize; 3]) -> PathBuf {
    ZarrEnvironment::create(root, input, chunk).unwrap();
    root.join("level0")
}

// --------------------------------------------------------------- the run --

/// **The acceptance criterion.** The same chain, over the same data, once from
/// memory and once bound to an array already on the disk.
#[test]
fn a_chain_over_an_attached_array_answers_what_the_same_chain_answers_in_memory() {
    let input = intensities(VOLUME);
    let workflow = median_workflow(VOLUME);
    let decomposition = plan(&workflow, VOLUME, 7);

    let memory = ArrayEnvironment::for_decomposition(input.clone(), &decomposition, [8, 8, 8])
        .unwrap();
    execute(
        "memory",
        &workflow,
        &decomposition,
        &Hints::default(),
        &memory,
    )
    .unwrap();
    let expected = memory.output();

    let source = Scratch::new("source");
    let array = stored(source.path(), &input, [5, 5, 5]);

    let work = Scratch::new("work");
    let env = ZarrEnvironment::attach(work.path(), &[AttachedImage::at(&array)]).unwrap();
    execute("attached", &workflow, &decomposition, &Hints::default(), &env).unwrap();

    assert_same(&expected, &env.output().unwrap(), "attached against memory");
}

/// The shape, element type and chunking come off the array, not from the
/// caller. Nothing here restates them, so nothing here can restate them wrongly.
#[test]
fn an_attached_array_describes_itself() {
    let input = intensities(VOLUME);
    let source = Scratch::new("describes");
    let array = stored(source.path(), &input, [4, 5, 6]);

    let work = Scratch::new("describes-work");
    let env = ZarrEnvironment::attach(work.path(), &[AttachedImage::at(&array)]).unwrap();

    assert_eq!(env.volume(), VOLUME);
    assert_eq!(env.image_shape(0).unwrap(), VOLUME);
    assert_eq!(env.image_dtype(0).unwrap(), Dtype::F64);
    assert_eq!(env.chunk_at(0).unwrap(), [4, 5, 6]);
    assert_same(&input, &env.image(0).unwrap(), "attached image 0");
}

// ---------------------------------------------------------- the window --

/// One plane of a multi-plane array is an image of its own — which is how a
/// channel of an OME-Zarr `[c, y, x]` level is read without copying it out.
#[test]
fn a_window_of_one_plane_reads_that_plane_and_no_other() {
    let stack = [4, 20, 24];
    let input = intensities(stack);
    let source = Scratch::new("plane");
    let array = stored(source.path(), &input, [1, 5, 6]);

    for channel in 0..stack[0] {
        let work = Scratch::new("plane-work");
        let env = ZarrEnvironment::attach(
            work.path(),
            &[AttachedImage::at(&array).plane(channel, [stack[1], stack[2]])],
        )
        .unwrap();

        assert_eq!(env.volume(), [1, stack[1], stack[2]]);

        let read = env.image(0).unwrap();
        let expected = input.view::<f64>().unwrap();
        let got = read.view::<f64>().unwrap();
        for y in 0..stack[1] {
            for x in 0..stack[2] {
                assert_eq!(
                    got[[0, y, x]],
                    expected[[channel, y, x]],
                    "channel {channel} at ({y}, {x})"
                );
            }
        }
    }
}

/// A window is applied to every read, not only to the one that reads the whole
/// image — so a block in the middle of the image lands where the window says.
///
/// This is the offset-applied-in-one-place-and-forgotten-in-another failure,
/// and it is invisible to the test above: reading the whole image starts at
/// `[0, 0, 0]`, where an offset that is dropped and an offset that is applied
/// look the same.
#[test]
fn a_window_moves_every_read_and_not_only_the_whole_one() {
    let volume = [8, 20, 24];
    let input = intensities(volume);
    let source = Scratch::new("region");
    let array = stored(source.path(), &input, [4, 5, 6]);

    let start = [3, 7, 9];
    let shape = [2, 6, 5];
    let work = Scratch::new("region-work");
    let env = ZarrEnvironment::attach(
        work.path(),
        &[AttachedImage::at(&array).window(start, shape)],
    )
    .unwrap();

    // A sub-box of the window, deliberately not at its origin and deliberately
    // not on the chunk grid.
    let region = Region::new(&[1, 2, 1], &[1, 3, 3]);
    let block = env.read(0, &region).unwrap();
    let got = block.as_array().unwrap().view::<f64>().unwrap();
    let expected = input.view::<f64>().unwrap();
    for i in 0..region.shape[0] {
        for j in 0..region.shape[1] {
            for k in 0..region.shape[2] {
                assert_eq!(
                    got[[i, j, k]],
                    expected[[
                        start[0] + region.start[0] + i,
                        start[1] + region.start[1] + j,
                        start[2] + region.start[2] + k
                    ]],
                    "window offset at ({i}, {j}, {k})"
                );
            }
        }
    }
}

/// A run over a window is a run over that sub-box and nothing else: the same
/// chain over the window's own data, in memory, gives the same answer.
///
/// The interesting part is what it does *not* say — a filter at the window's
/// edge reads what the window makes available, so the window's edge behaves
/// like a volume edge. That is the definition rather than a defect, and it is
/// why the reference here is the cut-out volume rather than the same voxels of
/// the whole one.
#[test]
fn a_run_over_a_window_is_a_run_over_that_sub_box() {
    let volume = [8, 20, 24];
    let input = intensities(volume);
    let start = [2, 4, 6];
    let shape = [4, 10, 12];

    let cut = {
        let view = input.view::<f64>().unwrap();
        let mut array = Array3::zeros((shape[0], shape[1], shape[2]));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    array[[i, j, k]] = view[[start[0] + i, start[1] + j, start[2] + k]];
                }
            }
        }
        Voxels::from(array)
    };

    let workflow = median_workflow(shape);
    let decomposition = plan(&workflow, shape, 5);

    let memory =
        ArrayEnvironment::for_decomposition(cut.clone(), &decomposition, [8, 8, 8]).unwrap();
    execute(
        "memory",
        &workflow,
        &decomposition,
        &Hints::default(),
        &memory,
    )
    .unwrap();

    let source = Scratch::new("window-run");
    let array = stored(source.path(), &input, [4, 5, 6]);
    let work = Scratch::new("window-run-work");
    let env = ZarrEnvironment::attach(
        work.path(),
        &[AttachedImage::at(&array).window(start, shape)],
    )
    .unwrap();
    execute("windowed", &workflow, &decomposition, &Hints::default(), &env).unwrap();

    assert_same(
        &memory.output(),
        &env.output().unwrap(),
        "windowed run against the cut-out volume",
    );
}

// ------------------------------------------------------------ refusals --

/// A supplied input is read at the reading block's own fetch region, so it is
/// in image 0's coordinate space and no other. One that is not is refused when
/// the environment is built, naming both extents.
#[test]
fn a_supplied_input_of_a_different_extent_is_refused_by_name() {
    let source = Scratch::new("extent");
    let first = Scratch::new("extent-a");
    let second = Scratch::new("extent-b");
    let _ = &source;

    let image0 = stored(first.path(), &intensities(VOLUME), [4, 4, 4]);
    let other = stored(second.path(), &intensities([8, 20, 24]), [4, 4, 4]);

    let work = Scratch::new("extent-work");
    let error = refusal(ZarrEnvironment::attach(
        work.path(),
        &[AttachedImage::at(&image0), AttachedImage::at(&other)],
    ));

    assert!(error.contains("coordinate space"), "got: {error}");
    assert!(error.contains("[8, 20, 24]"), "got: {error}");
}

/// A window that runs off the end of the array is refused where it is stated,
/// rather than at the first read that falls outside.
#[test]
fn a_window_that_does_not_fit_is_refused() {
    let source = Scratch::new("fit");
    let array = stored(source.path(), &intensities(VOLUME), [4, 4, 4]);
    let work = Scratch::new("fit-work");

    let error = refusal(ZarrEnvironment::attach(
        work.path(),
        &[AttachedImage::at(&array).window([0, 0, 20], [16, 20, 8])],
    ));
    assert!(error.contains("does not fit"), "got: {error}");

    let empty = refusal(ZarrEnvironment::attach(
        work.path(),
        &[AttachedImage::at(&array).window([0, 0, 0], [16, 0, 24])],
    ));
    assert!(empty.contains("does not fit"), "got: {empty}");
}

/// A run has an image 0. An empty list says so rather than producing an
/// environment with nothing to read.
#[test]
fn attaching_nothing_is_refused() {
    let work = Scratch::new("nothing");
    let error = refusal(ZarrEnvironment::attach(work.path(), &[]));
    assert!(error.contains("no images"), "got: {error}");
}
