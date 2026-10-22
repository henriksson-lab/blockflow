// SPDX-License-Identifier: MIT
//! Scale-controlled voxel envelopes using exterior rolling-ball accessibility.
//!
//! A center is admissible when its physical Euclidean distance from every
//! foreground voxel center is **strictly greater** than `radius`. Retain only
//! admissible centers connected to an array boundary, then remove background
//! within `radius` of those centers. The complement is the envelope. Thus a
//! ball can enter a wide mouth but cannot carve through a narrow passage.
//!
//! This is a discrete rolling-ball construction, not an alpha shape, CGAL alpha
//! wrap, convexity guarantee, or tissue segmentation. Enclosed voids are filled.
//! The input is always contained, but separate input objects need not be joined.
//! Keep the original mask for material support; the envelope may bridge gaps.
//!
//! The finite array is the domain: only centers inside it are considered, and
//! all nonsingleton faces seed the exterior. Pad with known background by more
//! than the radius if a free-space exterior is intended. Cropped/touching tissue
//! or a radius too large for the padding can make the envelope reach the edge.
//! Singleton axes are ignored for exterior access (2-D planes are supported).
//! Radius zero is ordinary hole filling with the selected background connectivity.
//!
//! The planned path composes existing exact distance sweeps and global hole
//! filling, so a block never runs its own independent flood. The resident helper
//! is for small/coarse arrays and tests; it allocates whole-volume intermediates.

use super::components::Connectivity;
use super::distance::{self, DistanceParams, Unbounded};
use super::fill;
use super::voxelwise::{VoxelwiseMaskOp, WidenOp};
use crate::assemble::{Phase, PlanBuilder};
use crate::dtype::Dtype;
use crate::error::{ensure, Result};
use crate::op::Chain;
use crate::sidecar::Lifecycle;
use ndarray::{Array3, ArrayView3};

/// Radius and per-axis z,y,x pitch share any caller-selected length unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RollingBallEnvelope {
    radius: f64,
    sampling: [f64; 3],
    connectivity: Connectivity,
}

impl RollingBallEnvelope {
    pub fn new(radius: f64, sampling: [f64; 3]) -> Result<Self> {
        ensure!(
            radius.is_finite() && radius >= 0.,
            "envelope: radius must be finite and non-negative"
        );
        for pitch in sampling {
            ensure!(
                pitch.is_finite()
                    && pitch > 0.
                    && (pitch * pitch).is_finite()
                    && pitch * pitch > 0.,
                "envelope: pitch and its square must be finite and positive"
            );
        }
        Ok(Self {
            radius,
            sampling,
            connectivity: Connectivity::Faces,
        })
    }
    /// Connectivity of admissible exterior ball centers, not of tissue.
    pub fn connecting(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }
    pub fn radius(self) -> f64 {
        self.radius
    }
    pub fn sampling(self) -> [f64; 3] {
        self.sampling
    }
    pub fn connectivity(self) -> Connectivity {
        self.connectivity
    }
    fn distance(self) -> DistanceParams {
        DistanceParams::default()
            .with_sampling(self.sampling)
            .with_unbounded(Unbounded::Infinite)
    }
    fn validate_shape(self, shape: [usize; 3]) -> Result<()> {
        ensure!(
            shape.iter().all(|&n| n > 0) && shape.iter().any(|&n| n > 1),
            "envelope: nonempty shape with at least one nonsingleton axis required"
        );
        let diagonal_squared: f64 = (0..3)
            .map(|i| ((shape[i] - 1) as f64 * self.sampling[i]).powi(2))
            .sum();
        ensure!(
            diagonal_squared.is_finite(),
            "envelope: physical diagonal overflows distance arithmetic"
        );
        Ok(())
    }
}

/// Resident reference using the same distance and component primitives as the plan.
/// No outside-array tissue/background is invented; see the module domain contract.
pub fn rolling_ball_envelope(
    mask: ArrayView3<'_, bool>,
    config: RollingBallEnvelope,
) -> Result<Array3<bool>> {
    let shape = [mask.shape()[0], mask.shape()[1], mask.shape()[2]];
    config.validate_shape(shape)?;
    let complement = mask.mapv(|v| !v);
    let distance = distance::distance_transform(complement.view(), &config.distance())?;
    let blocked = distance.mapv(|d| d <= config.radius);
    drop(distance);
    drop(complement);
    let mut labels = Array3::<u32>::zeros(mask.raw_dim());
    let count =
        fill::label_background_into_with(blocked.view(), config.connectivity, labels.view_mut())?;
    let flags = fill::outside_flags(labels.view(), count, [0; 3], shape, shape);
    let mut not_exterior = blocked;
    fill::fill_from_labels_into(labels.view(), &flags, not_exterior.view_mut())?;
    drop(labels);
    let distance = distance::distance_transform(not_exterior.view(), &config.distance())?;
    Ok(distance.mapv(|d| d > config.radius))
}

/// Append a bool-mask -> bool-envelope pipeline and return its final phase.
///
/// `block` is the transverse tile edge for whole-axis distance sweeps and the
/// hole-filling grid. `stream` names a unique component-fragment stream. The
/// helper leaves the plan on the distance helper's `[block;3]` grid. No global
/// pixel array is pooled in a reduce; the hole fill uses its normal seam merge.
/// The caller chooses execution concurrency/memory budgets as for other plans.
pub fn append_rolling_ball_envelope_phases(
    plan: &mut PlanBuilder,
    config: RollingBallEnvelope,
    block: usize,
    stream: impl Into<String>,
    lifecycle: Lifecycle,
) -> Result<Phase> {
    config.validate_shape(plan.grid().volume())?;
    ensure!(
        plan.reads() == Dtype::Bool,
        "envelope: plan input must be Bool"
    );
    ensure!(block > 0, "envelope: block edge must be positive");
    let stream = stream.into();
    crate::sidecar::check_stream_name(&stream)?;
    plan.pixels(Chain::sequence(vec![
        Chain::op(WidenOp::new("envelope mask widen")),
        Chain::op(VoxelwiseMaskOp::new("envelope mask complement", |v| {
            v == 0.
        })),
    ]))?;
    distance::append_to(plan, &config.distance(), block)?;
    let radius = config.radius;
    plan.pixels(Chain::op(VoxelwiseMaskOp::new(
        "envelope blocked centers",
        move |v| v <= radius,
    )))?;
    fill::append_connected(plan, stream, lifecycle, Dtype::Bool, config.connectivity)?;
    distance::append_to(plan, &config.distance(), block)?;
    plan.pixels(Chain::op(VoxelwiseMaskOp::threshold(
        "rolling-ball envelope",
        radius,
    )))
}
