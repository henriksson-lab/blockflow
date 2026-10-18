// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// **A declared background changes the storage, never the answer.**
//
// An image whose phase declares a background (`PhaseDecomposition::background`)
// is stored with it as the fill value. Three things follow, and each is checked
// here against the same plan run without the declaration and against memory:
//
// 1. the voxels are identical, bit for bit;
// 2. a chunk that is all background is not written;
// 3. a block whose chunks are all absent is short-circuited before it is read,
//    and a short-circuited block equal to the background is not written.

#![cfg(feature = "zarr")]

use std::path::{Path, PathBuf};

use ndarray::Array3;

use blockflow::assemble::PlanBuilder;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::SmoothOp;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::Dtype;

const VOLUME: [usize; 3] = [48, 48, 64];
const BLOCK: [usize; 3] = [16, 16, 16];

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("blockflow-background-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// Zero everywhere but one bright, uneven box near a corner.
fn sparse_input() -> Array3<u16> {
    Array3::from_shape_fn((VOLUME[0], VOLUME[1], VOLUME[2]), |(z, y, x)| {
        if (4..12).contains(&z) && (5..13).contains(&y) && (3..14).contains(&x) {
            100 + ((z * 7 + y * 3 + x) % 11) as u16
        } else {
            0
        }
    })
}

fn chunk_files(level: &Path) -> usize {
    fn walk(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| {
                        let path = entry.path();
                        if path.is_dir() {
                            walk(&path)
                        } else {
                            usize::from(path.file_name().is_some_and(|n| n != "zarr.json"))
                        }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }
    walk(level)
}

struct Run {
    output: Array3<f64>,
    stats: blockflow::log::Stats,
    level1_chunks: usize,
    read_voxels: u64,
}

/// Two smoothing phases; the second reads what the first wrote. With
/// `background`, image 0 is declared to be mostly 0, which `SmoothOp` maps to 0.
fn run(name: &str, background: bool) -> Run {
    let root = scratch(name);
    let input = Voxels::from(sparse_input());
    let env = ZarrEnvironment::create(&root, &input, BLOCK).unwrap();
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U16, BlockGrid::new(VOLUME, BLOCK).unwrap());
    if background {
        builder = builder.with_input_background(0.0);
    }
    builder.pixels(Chain::op(SmoothOp::isotropic("first", 1.0, 3.0).unwrap())).unwrap();
    builder.pixels(Chain::op(SmoothOp::isotropic("second", 1.0, 3.0).unwrap())).unwrap();
    let assembly = builder.finish().unwrap();
    assert_eq!(assembly.decomposition.background_at(1), background.then_some(0.0));
    let hints = Hints {
        concurrency: 4,
        keep_images: [blockflow::assemble::ImageId::from(1)].into_iter().collect(),
        ..Hints::default()
    };
    let stats = execute_phases(
        "background",
        &assembly.workflow,
        &assembly.decomposition,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )
    .unwrap();
    let output = env.output().unwrap().view::<f64>().unwrap().to_owned();
    let level1_chunks = chunk_files(&root.join("level1"));
    let read_voxels = env.counters().read_voxels.load(std::sync::atomic::Ordering::SeqCst);
    let _ = std::fs::remove_dir_all(&root);
    Run { output, stats, level1_chunks, read_voxels }
}

#[test]
fn a_background_saves_chunks_and_reads_and_changes_no_voxel() {
    let plain = run("plain", false);
    let sparse = run("sparse", true);

    // the oracle: the same plan in memory
    let mut builder = PlanBuilder::new(VOLUME, Dtype::U16, BlockGrid::new(VOLUME, BLOCK).unwrap());
    builder.pixels(Chain::op(SmoothOp::isotropic("first", 1.0, 3.0).unwrap())).unwrap();
    builder.pixels(Chain::op(SmoothOp::isotropic("second", 1.0, 3.0).unwrap())).unwrap();
    let assembly = builder.finish().unwrap();
    let memory =
        ArrayEnvironment::for_decomposition(Voxels::from(sparse_input()), &assembly.decomposition, BLOCK).unwrap();
    execute_phases("memory", &assembly.workflow, &assembly.decomposition, &Hints::default(), &memory, &[], &assembly.work())
        .unwrap();
    let expected = memory.output().view::<f64>().unwrap().to_owned();

    assert!(expected.iter().any(|v| *v != 0.0), "the oracle is not all background");
    for (name, got) in [("plain", &plain.output), ("sparse", &sparse.output)] {
        let differing = got.iter().zip(expected.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(differing, 0, "{name}: {differing} voxels differ from memory");
    }

    // every chunk of image 1 written without a background; only the few
    // around the box with one
    let total = VOLUME.iter().zip(BLOCK).map(|(v, b)| v.div_ceil(b)).product::<usize>();
    assert_eq!(plain.level1_chunks, total);
    assert!(sparse.level1_chunks > 0 && sparse.level1_chunks < total / 4, "{} of {total}", sparse.level1_chunks);

    // the same blocks are uniform either way; with a background the second
    // phase finds them so without reading them
    assert!(plain.stats.tasks_short_circuited > 0);
    assert_eq!(sparse.stats.tasks_short_circuited, plain.stats.tasks_short_circuited);
    assert!(sparse.read_voxels < plain.read_voxels, "{} vs {}", sparse.read_voxels, plain.read_voxels);
    eprintln!(
        "chunks of image 1: {} vs {}; voxels read: {} vs {}; blocks short-circuited: {}",
        sparse.level1_chunks, plain.level1_chunks, sparse.read_voxels, plain.read_voxels, sparse.stats.tasks_short_circuited
    );
}
