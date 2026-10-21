// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// The per-block dirty set of an iterative phase (`IterativeOp::stationary`),
// held against the full sweep it replaces. The claim is that it changes the
// work and nothing else: the same voxels and the same substage count, on every
// grid, with fewer substage calls. The fixtures are the two shapes an iteration
// takes in practice — a thinning that does most of its work at once and then
// a long thin tail, and a propagation along a one-voxel serpentine whose front
// crosses one seam at a time — over grids from one block to one block per few
// voxels, ragged ones included.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::Array3;

use blockflow::assemble::PlanBuilder;
use blockflow::dtype::Dtype;
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::{IterativeOp, Substage, SubstageLimit, SubstageOperand};
use blockflow::ops::directional::{
    clear_faces, directional_pass_into, directional_reach, directional_to_fixed_point,
};
use blockflow::ops::skeleton::PassLimit;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::Result;

/// One directional pass per substage, counting its calls.
struct Thinning {
    stationary: bool,
    calls: Arc<AtomicUsize>,
}

impl IterativeOp for Thinning {
    fn name(&self) -> &'static str {
        "thinning"
    }

    fn operands(&self) -> Vec<SubstageOperand> {
        vec![SubstageOperand::running([directional_reach(1); 3])]
    }

    fn limit(&self) -> SubstageLimit {
        SubstageLimit::of(1000).unwrap()
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::Bool
    }

    fn stationary(&self) -> bool {
        self.stationary
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        directional_pass_into(at.operand(0)?.view::<bool>()?, out.view_mut::<bool>()?)
    }
}

/// `g <- g or (any face neighbour of g, and f)`: the running set grows one
/// voxel per substage along the fixed mask `f`.
struct Propagation {
    stationary: bool,
    calls: Arc<AtomicUsize>,
}

impl IterativeOp for Propagation {
    fn name(&self) -> &'static str {
        "propagation"
    }

    fn operands(&self) -> Vec<SubstageOperand> {
        vec![
            SubstageOperand::running([1, 1, 1]),
            SubstageOperand::source(blockflow::assemble::ImageId::supplied(0), [0, 0, 0]),
        ]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<blockflow::op::SourceInput> {
        vec![
            blockflow::op::SourceInput::voxelwise(blockflow::assemble::ImageId::supplied(0))
                .holding(Dtype::Bool),
        ]
    }

    fn limit(&self) -> SubstageLimit {
        SubstageLimit::of(5000).unwrap()
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::Bool
    }

    fn stationary(&self) -> bool {
        self.stationary
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let g = at.operand(0)?.view::<bool>()?;
        let f = at.operand(1)?.view::<bool>()?;
        let mut out = out.view_mut::<bool>()?;
        let shape = [g.shape()[0], g.shape()[1], g.shape()[2]];
        for ((i, j, k), slot) in out.indexed_iter_mut() {
            let mut grown = g[[i, j, k]];
            if !grown && f[[i, j, k]] {
                let position = [i, j, k];
                for axis in 0..3 {
                    for step in [-1isize, 1] {
                        let mut p = position;
                        let moved = p[axis] as isize + step;
                        if moved < 0 || moved >= shape[axis] as isize {
                            continue;
                        }
                        p[axis] = moved as usize;
                        grown |= g[p];
                    }
                }
            }
            *slot = grown;
        }
        Ok(())
    }
}

fn run(
    op: impl IterativeOp + 'static,
    input: Array3<bool>,
    supplied: Vec<Voxels>,
    block: [usize; 3],
) -> (Array3<bool>, usize) {
    let volume = [input.shape()[0], input.shape()[1], input.shape()[2]];
    let mut plan = PlanBuilder::new(volume, Dtype::Bool, BlockGrid::new(volume, block).unwrap());
    let written = plan.iterate(op).unwrap().image().unwrap();
    let assembly = plan.finish().unwrap();
    let input: Voxels = input.into();
    let env = match supplied.is_empty() {
        true => ArrayEnvironment::for_decomposition(input, &assembly.decomposition, [8, 8, 8]),
        false => ArrayEnvironment::with_inputs(input, supplied, &assembly.decomposition, [8, 8, 8]),
    }
    .unwrap();
    let hints = Hints {
        concurrency: 4,
        keep_images: [written].into_iter().collect::<BTreeSet<_>>(),
        ..Hints::default()
    };
    let stats = execute_phases(
        "dirty set",
        &assembly.workflow,
        &assembly.decomposition,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )
    .unwrap();
    let out = env.try_image(written.index()).unwrap();
    let substages = stats.substages.iter().copied().sum();
    (out.view::<bool>().unwrap().to_owned(), substages)
}

fn tubes(shape: [usize; 3]) -> Array3<bool> {
    // Two crossing tubes and a slab: thick parts that take several passes,
    // thin parts that are done at once, and a crossing that keeps a tail.
    let mut volume = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        let (z, y, x) = (i as f64, j as f64, k as f64);
        let a = (y - 10.0).powi(2) + (z - 8.0).powi(2) < 20.0;
        let b = (x - 17.0).powi(2) + (z - 9.0).powi(2) < 12.0;
        let slab = (3..6).contains(&i) && (25..33).contains(&j) && (4..30).contains(&k);
        a || b || slab
    });
    clear_faces(&mut volume);
    volume
}

const GRIDS: [[usize; 3]; 6] = [
    [16, 40, 36],
    [8, 20, 18],
    [8, 8, 8],
    [5, 7, 6],
    [3, 4, 5],
    [16, 13, 36],
];

#[test]
fn the_dirty_set_thins_to_the_same_skeleton_in_the_same_substages() {
    let input = tubes([16, 40, 36]);
    let (whole, passes) =
        directional_to_fixed_point(input.view(), PassLimit::of(1000).unwrap()).unwrap();
    for block in GRIDS {
        let full = Thinning {
            stationary: false,
            calls: Arc::default(),
        };
        let (swept, swept_substages) = run(full, input.clone(), Vec::new(), block);
        let dirty = Thinning {
            stationary: true,
            calls: Arc::default(),
        };
        let (skipped, skipped_substages) = run(dirty, input.clone(), Vec::new(), block);
        assert_eq!(swept, whole, "the full sweep on {block:?}");
        assert_eq!(skipped, whole, "the dirty set on {block:?}");
        assert_eq!(swept_substages, passes, "{block:?}");
        assert_eq!(skipped_substages, passes, "{block:?}");
    }
}

#[test]
fn the_dirty_set_runs_fewer_substages_where_blocks_settle() {
    let input = tubes([16, 40, 36]);
    let block = [8, 8, 8];
    let blocks = BlockGrid::new([16, 40, 36], block).unwrap().n_blocks();
    let (swept, visited) = (Arc::<AtomicUsize>::default(), Arc::<AtomicUsize>::default());
    let full = Thinning {
        stationary: false,
        calls: swept.clone(),
    };
    let dirty = Thinning {
        stationary: true,
        calls: visited.clone(),
    };
    let (a, substages) = run(full, input.clone(), Vec::new(), block);
    let (b, _) = run(dirty, input, Vec::new(), block);
    assert_eq!(a, b);
    let (swept, visited) = (swept.load(Ordering::SeqCst), visited.load(Ordering::SeqCst));
    assert_eq!(
        swept,
        blocks * substages,
        "the full sweep visits every block"
    );
    assert!(
        visited < swept,
        "the dirty set visited {visited} blocks of {swept}"
    );
}

#[test]
fn a_front_along_a_serpentine_crosses_the_seams_in_the_same_substages() {
    // A one-voxel serpentine in one plane, its start seeded: the front moves
    // one voxel per substage, so the iteration is long and at any substage only
    // the blocks around the front change.
    let shape = [3usize, 21, 22];
    let path = Array3::from_shape_fn((shape[0], shape[1], shape[2]), |(i, j, k)| {
        // Rows at `j % 4 == 1`, joined alternately at their right and left
        // ends by the three voxels between them.
        let joint = match ((j.max(1) - 1) / 4) % 2 {
            0 => 20,
            _ => 1,
        };
        i == 1 && (1..=20).contains(&k) && j >= 1 && (j % 4 == 1 || k == joint)
    });
    let mut seed = Array3::from_elem((shape[0], shape[1], shape[2]), false);
    seed[[1, 1, 1]] = true;
    let mut reference: Option<(Array3<bool>, usize)> = None;
    for block in [[3, 21, 22], [3, 5, 5], [1, 3, 4], [2, 7, 6]] {
        for stationary in [false, true] {
            let op = Propagation {
                stationary,
                calls: Arc::default(),
            };
            let got = run(op, seed.clone(), vec![path.clone().into()], block);
            match &reference {
                None => {
                    assert!(got.1 > 50, "the front took {} substages", got.1);
                    reference = Some(got);
                }
                Some(expected) => assert_eq!(&got, expected, "{block:?}, {stationary}"),
            }
        }
    }
}
