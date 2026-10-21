// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The skeleton: the mask's six faces cleared, then thinned to a fixed point by
//! `blockflow::ops::directional` passes.
//!
//! One thinning pass is twelve directional sub-iterations and reaches twelve
//! voxels; running to convergence would grow that halo without bound, so the
//! passes run as an `IterativeOp` instead. Each substage is one round of
//! passes at a fixed halo, neighbours exchange their previous round through
//! that halo, and the phase stops when no block changed anywhere — the global
//! convergence test a blocked thinning needs, since a voxel deletable in one
//! block can make a neighbour's voxel deletable in the next round. The op
//! declares itself stationary, so a round runs only the blocks a change in the
//! round before could reach (the executor's dirty set); the rounds and the
//! skeleton are the same.
//!
//! The faces are cleared because the thinning templates treat the outside of
//! the volume as background only when the volume's own faces hold none of the
//! object; a vessel cut by the volume face would otherwise thin differently
//! depending on where the face fell.

use blockflow::iterate::{IterativeOp, Substage, SubstageLimit, SubstageOperand};
use blockflow::op::{Anchor, BlockOp, Chain};
use blockflow::ops::directional::{
    directional_passes_into, directional_reach, DIRECTIONAL_PASS_COST,
};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Error, Result};

/// How the skeleton is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletonParams {
    /// Clear the volume's six faces before thinning.
    pub delete_border: bool,
    /// Directional passes per round. More passes per round means fewer rounds
    /// (fewer halo exchanges) at a wider halo: `12 * passes` voxels.
    pub passes_per_round: usize,
    /// Rounds before the run is refused as not converging.
    pub round_limit: usize,
}

impl Default for SkeletonParams {
    fn default() -> Self {
        Self {
            delete_border: true,
            passes_per_round: 1,
            round_limit: 4096,
        }
    }
}

impl SkeletonParams {
    pub fn validate(&self) -> Result<()> {
        if self.passes_per_round == 0 || self.round_limit == 0 {
            return Err(Error::invalid(
                "skeleton: a round of zero passes, or a limit of zero rounds, thins nothing",
            ));
        }
        Ok(())
    }

    /// The thinning, as the iterative op a `PlanBuilder::iterate` phase runs.
    pub fn thinning(&self) -> Result<ThinToFixedPoint> {
        Ok(ThinToFixedPoint {
            passes: self.passes_per_round,
            limit: SubstageLimit::of(self.round_limit)?,
        })
    }
}

/// Clear every voxel on the volume's six faces. Reach zero: whether a voxel is
/// on a face is a fact about its global position, which [`Anchor`] gives.
pub struct ClearFacesOp;

impl BlockOp for ClearFacesOp {
    fn name(&self) -> &'static str {
        "clear faces"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::Bool
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, at: &Anchor) -> Result<()> {
        let input = input.view::<bool>()?;
        let mut out = out.view_mut::<bool>()?;
        if input.shape() != out.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: out.shape().to_vec(),
            });
        }
        for ((i, j, k), slot) in out.indexed_iter_mut() {
            let global = [at.offset[0] + i, at.offset[1] + j, at.offset[2] + k];
            let on_face =
                (0..3).any(|axis| global[axis] == 0 || global[axis] + 1 == at.volume[axis]);
            *slot = input[[i, j, k]] && !on_face;
        }
        Ok(())
    }

    /// A block of background stays background; a set block does not stay set
    /// where it touches a face.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }
}

/// Directional thinning to a fixed point, `passes` passes per round.
pub struct ThinToFixedPoint {
    passes: usize,
    limit: SubstageLimit,
}

impl IterativeOp for ThinToFixedPoint {
    fn name(&self) -> &'static str {
        "thin to fixed point"
    }

    fn operands(&self) -> Vec<SubstageOperand> {
        vec![SubstageOperand::running(
            [directional_reach(self.passes); 3],
        )]
    }

    fn limit(&self) -> SubstageLimit {
        self.limit
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::Bool
    }

    /// A round is the same passes at every substage, a function of the window
    /// alone, so the executor revisits only the blocks whose window changed in
    /// the previous round: after the first few rounds the thinning is a
    /// handful of blocks along the last thick parts, not the whole volume.
    fn stationary(&self) -> bool {
        true
    }

    fn substage(&self, at: &Substage<'_>, out: &mut Voxels) -> Result<()> {
        directional_passes_into(
            at.operand(0)?.view::<bool>()?,
            self.passes,
            out.view_mut::<bool>()?,
        )
    }

    fn cost_per_voxel(&self) -> f64 {
        DIRECTIONAL_PASS_COST * self.passes as f64
    }
}

/// The face clearing as a chain node, for the pixel phase before the thinning.
pub fn clear_faces_node() -> Chain {
    Chain::op(ClearFacesOp)
}
