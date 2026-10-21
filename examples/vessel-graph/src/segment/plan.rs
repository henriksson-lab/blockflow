// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Plan assembly shared by the segmentation and tract stages, and the in-memory
//! runner the library API and the tests use.
//!
//! A pixel phase is charged for its own input — the image the phase before it
//! wrote — whether or not its chain reads it, so a chain that names an image by
//! a `Chain::source` leaf while its input is that same image fetches it twice.
//! [`Cursor`] tracks which image the next phase is handed and spells a read
//! either way: a carry of the input, or a source leaf. The arms are ordered so
//! that most phases read their own input.

use std::collections::BTreeSet;

use blockflow::assemble::{Assembly, ImageId, Materialisation, Phase, PlanBuilder, Substages};
use blockflow::decomposition::{refined_ladder, Constraints};
use blockflow::env::ArrayEnvironment;
use blockflow::geometry::BlockGrid;
use blockflow::iterate::IterativeOp;
use blockflow::log::Stats;
use blockflow::op::Chain;
use blockflow::ops::voxelwise::{CarryOp, WidenOp};
use blockflow::strategy::{execute_phases, Enumerating, Hints};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Error, Result};

/// An image of the plan and the element type it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Image {
    pub id: ImageId,
    pub dtype: Dtype,
}

impl Image {
    pub fn new(id: impl Into<ImageId>, dtype: Dtype) -> Self {
        Self {
            id: id.into(),
            dtype,
        }
    }
}

/// Workers for planning and for runs: rayon's pool size.
pub fn workers() -> usize {
    rayon::current_num_threads().max(1)
}

/// A [`PlanBuilder`] plus the image its next pixel phase is handed.
pub struct Cursor<'a> {
    pub plan: &'a mut PlanBuilder,
    input: Image,
    workers: usize,
    /// Whether an iterative phase is given the block grid the planner prices
    /// fastest for it ([`Self::iterate`]) rather than the plan's.
    priced_iterations: bool,
    /// The grid label images are written on, when it is not the plan's
    /// ([`Self::to_label_grid`]).
    label_grid: Option<BlockGrid>,
}

/// The block edges an iterative phase is priced at when it chooses its own
/// grid: the refined ladder (a rung at three quarters of each octave) from 32
/// to 256, cut on every axis.
const ITERATE_LADDER: [usize; 4] = [32, 64, 128, 256];

impl<'a> Cursor<'a> {
    /// `input` is what the first appended phase reads: image 0 for a fresh plan.
    pub fn new(plan: &'a mut PlanBuilder, input: Image) -> Self {
        Self {
            plan,
            input,
            workers: workers(),
            priced_iterations: false,
            label_grid: None,
        }
    }

    /// Write label images on `grid` ([`Self::to_label_grid`]) rather than on
    /// the plan's grid. A stored image's chunk shape follows the block that
    /// wrote it, so this is what keeps a published label's layout apart from
    /// the grid the work ran on.
    pub fn labelling_on(mut self, grid: BlockGrid) -> Self {
        self.label_grid = Some(grid);
        self
    }

    /// Cut the phases from here on on the label grid, if one was given.
    pub fn to_label_grid(&mut self) {
        if let Some(grid) = &self.label_grid {
            self.plan.regrid(grid.clone());
        }
    }

    /// Give each iterative phase the grid the planner prices fastest for it
    /// on this cursor's workers (`PlanBuilder::iterate_priced`) instead of the
    /// plan's grid. An iteration's reach is usually unlike the pixel phases'
    /// around it — the thinning reaches as far on every axis, the windows
    /// before it far in the image plane and hardly at all across it — so the
    /// plan's grid is rarely the right one for it.
    pub fn pricing_iterations(mut self) -> Self {
        self.priced_iterations = true;
        self
    }

    pub fn input(&self) -> Image {
        self.input
    }

    /// The nodes that bring `wanted` into a chain: nothing when it is the phase's
    /// input, a source leaf otherwise.
    pub fn nodes(&self, wanted: Image) -> Vec<Chain> {
        if wanted.id == self.input.id {
            Vec::new()
        } else {
            vec![Chain::source(wanted.id, wanted.dtype)]
        }
    }

    /// [`Self::nodes`], widened to `f64` where the image is narrower.
    pub fn nodes_f64(&self, wanted: Image) -> Vec<Chain> {
        let mut nodes = self.nodes(wanted);
        if wanted.dtype != Dtype::F64 {
            nodes.push(Chain::op(WidenOp::new("widen")));
        }
        nodes
    }

    /// A chain reading `wanted` and applying `tail`.
    pub fn on(&self, wanted: Image, tail: Vec<Chain>) -> Chain {
        let mut nodes = self.nodes(wanted);
        nodes.extend(tail);
        chain_of(nodes)
    }

    /// A chain reading `wanted` as `f64` and applying `tail`.
    pub fn on_f64(&self, wanted: Image, tail: Vec<Chain>) -> Chain {
        let mut nodes = self.nodes_f64(wanted);
        nodes.extend(tail);
        chain_of(nodes)
    }

    /// Append one pixel phase; its image becomes the next phase's input.
    pub fn pixels(&mut self, chain: Chain) -> Result<Image> {
        let phase = self.plan.pixels(chain)?;
        self.wrote(phase)
    }

    /// Append a chain the planner may cut into several phases, on the plan
    /// grid's own block edges and along the axes the plan grid cuts.
    ///
    /// The axes matter as much as the edges: the candidates are scalar edges,
    /// and on a grid that leaves an axis whole (a slab grid, `[16, 512, 404]`
    /// on a `[128, 512, 404]` volume) an edge offered on every axis would be a
    /// cube of the slab's thickness, with a wide-reach op's halo on all six
    /// sides instead of on two.
    pub fn partition(&mut self, chain: Chain) -> Result<Image> {
        let grid = self.plan.grid();
        let (block, volume) = (grid.block(), grid.volume());
        let mut split_axes: Vec<usize> =
            (0..3).filter(|&axis| block[axis] < volume[axis]).collect();
        if split_axes.is_empty() {
            split_axes = vec![0, 1, 2];
        }
        let mut candidates: Vec<usize> = split_axes.iter().map(|&axis| block[axis]).collect();
        candidates.sort_unstable();
        candidates.dedup();
        let constraints = Constraints {
            expected_concurrency: self.workers,
            block_candidates: candidates,
            split_axes,
            ..Constraints::default()
        };
        let strategy = Enumerating {
            concurrency: self.workers,
            ..Enumerating::default()
        };
        let partition = self.plan.partition(chain, &strategy, &constraints)?;
        self.input = Image::new(partition.image(), self.plan.reads());
        Ok(self.input)
    }

    /// Append an iterative phase reading the cursor's input: on the plan's
    /// grid, or on its own priced grid after [`Self::pricing_iterations`].
    pub fn iterate(&mut self, op: impl IterativeOp + 'static) -> Result<Image> {
        let phase = match self.priced_iterations {
            false => self.plan.iterate(op)?,
            true => {
                let constraints = Constraints {
                    expected_concurrency: self.workers,
                    block_candidates: refined_ladder(&ITERATE_LADDER),
                    split_axes: vec![0, 1, 2],
                    ..Constraints::default()
                };
                self.plan
                    .iterate_priced(
                        op,
                        &constraints,
                        self.workers,
                        Materialisation::Intermediate,
                        Substages::Unknown,
                    )?
                    .phase()
            }
        };
        self.wrote(phase)
    }

    /// Record a phase appended directly on the builder (a fragment or iterate
    /// phase) that wrote pixels.
    pub fn wrote(&mut self, phase: Phase) -> Result<Image> {
        self.input = Image::new(phase.image()?, self.plan.reads());
        Ok(self.input)
    }

    /// Make `wanted` the next phase's input, copying it into a phase of its own
    /// if it is not already. For phases that cannot name a source (fragment and
    /// iterate phases read their input).
    pub fn make_input(&mut self, wanted: Image) -> Result<Image> {
        if wanted.id == self.input.id {
            return Ok(wanted);
        }
        self.pixels(Chain::sequence(vec![
            Chain::source(wanted.id, wanted.dtype),
            Chain::op(CarryOp::new("carry")),
        ]))
    }
}

/// A source leaf on `image`, widened to `f64` where it is narrower: for a
/// branch inside a sequence, where the stream is not the phase's input.
pub fn source_f64(image: Image) -> Vec<Chain> {
    let mut nodes = vec![Chain::source(image.id, image.dtype)];
    if image.dtype != Dtype::F64 {
        nodes.push(Chain::op(WidenOp::new("widen")));
    }
    nodes
}

/// One node as itself, several as a sequence, none as a carry of the input.
pub fn chain_of(mut nodes: Vec<Chain>) -> Chain {
    match nodes.len() {
        0 => Chain::op(CarryOp::new("carry")),
        1 => nodes.pop().expect("one node"),
        _ => Chain::sequence(nodes),
    }
}

/// Run an assembled plan over arrays held in memory: `inputs[0]` is image 0 and
/// the rest are the supplied images in order. `keep` are the images to read
/// back afterwards.
pub fn run_in_memory(
    assembly: &Assembly,
    mut inputs: Vec<Voxels>,
    keep: &[ImageId],
) -> Result<(Stats, ArrayEnvironment)> {
    if inputs.is_empty() {
        return Err(Error::invalid("a run needs at least one input volume"));
    }
    let first = inputs.remove(0);
    let chunk = [32, 32, 32];
    let env = if inputs.is_empty() {
        ArrayEnvironment::for_decomposition(first, &assembly.decomposition, chunk)?
    } else {
        ArrayEnvironment::with_inputs(first, inputs, &assembly.decomposition, chunk)?
    };
    let hints = Hints {
        concurrency: workers(),
        keep_images: keep.iter().copied().collect::<BTreeSet<_>>(),
        ..Hints::default()
    };
    let stats = execute_phases(
        "vessel-graph",
        &assembly.workflow,
        &assembly.decomposition,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )?;
    Ok((stats, env))
}

/// A kept image of a finished in-memory run.
pub fn read_back(env: &ArrayEnvironment, image: Image) -> Result<Voxels> {
    env.try_image(image.id.index())
}
