// SPDX-License-Identifier: MIT
// Original work for this crate.

//! A density volume from a point set: every cell adds `1` (or its weight) to
//! each voxel of a ball around it (`ops::voxelize`), so a voxel holds the
//! number of cells (or the summed weight) within the ball's radius. Mass that
//! hangs off the volume is dropped, so density falls off within one radius of
//! a face.

use atlas_regions::{voxel_index, OutOfBounds, Rounding};
use blockflow::env::ArrayEnvironment;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::ops::voxelize::{ball, VoxelizeOp};
use blockflow::points::{Point, PointSourceOp};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, PlanBuilder};

const POINTS: &str = "density.points";

/// Image voxel positions mapped onto a density lattice `factor` times coarser
/// per axis, rounded to the nearest voxel centre; points that land outside
/// `volume` are dropped and counted.
pub fn place(points: &[[f64; 3]], factor: [f64; 3], volume: [usize; 3]) -> (Vec<Point>, usize) {
    let mut placed = Vec::with_capacity(points.len());
    let mut dropped = 0;
    for point in points {
        let mut at = [0usize; 3];
        let mut inside = true;
        for axis in 0..3 {
            let scaled = ((point[axis] + 0.5) / factor[axis] - 0.5).round_ties_even();
            if !(scaled >= 0.0) || scaled >= volume[axis] as f64 {
                inside = false;
                break;
            }
            at[axis] = scaled as usize;
        }
        if inside {
            placed.push(Point::unit(at));
        } else {
            dropped += 1;
        }
    }
    (placed, dropped)
}

/// Points already in density-lattice voxel coordinates, each picking its
/// voxel by `rounding`, carrying `1` or its entry of `weights`; points outside
/// `volume` (or with a non-finite weight) are dropped and counted.
pub fn place_at(
    points: &[[f64; 3]],
    weights: Option<&[f64]>,
    rounding: Rounding,
    volume: [usize; 3],
) -> Result<(Vec<Point>, usize)> {
    if let Some(weights) = weights {
        if weights.len() != points.len() {
            return Err(Error::invalid(format!(
                "{} weights for {} points",
                weights.len(),
                points.len()
            )));
        }
    }
    let mut placed = Vec::with_capacity(points.len());
    let mut dropped = 0;
    for (index, point) in points.iter().enumerate() {
        let weight = weights.map_or(1.0, |weights| weights[index]);
        match voxel_index(*point, volume, rounding, OutOfBounds::Invalid) {
            Some(at) if weight.is_finite() => placed.push(Point::weighted(at, weight)),
            _ => dropped += 1,
        }
    }
    Ok((placed, dropped))
}

/// Render `points` (density-lattice voxels) into an `f64` volume, with a ball
/// of `radius` voxels per axis, planned over `block`s.
pub fn density(
    points: Vec<Point>,
    volume: [usize; 3],
    radius: [usize; 3],
    block: [usize; 3],
) -> Result<Voxels> {
    if let Some(point) = points
        .iter()
        .find(|point| (0..3).any(|axis| point.at[axis] >= volume[axis]))
    {
        return Err(Error::invalid(format!(
            "the point at {:?} is outside the {volume:?} density volume",
            point.at
        )));
    }
    let grid = BlockGrid::new(volume, block)?;
    let mut plan = PlanBuilder::new(volume, Dtype::F64, grid.clone());
    let source = plan.fragments(PointSourceOp::new(
        "density points",
        POINTS,
        Lifecycle::DeleteOnExit,
        points,
    ))?;
    plan.fragments(VoxelizeOp::new(
        "density",
        POINTS,
        source.index(),
        ball(radius),
        &grid,
    )?)?;
    let assembly = plan.finish()?;
    let env = ArrayEnvironment::for_decomposition(
        Voxels::zeros(Dtype::F64, volume)?,
        &assembly.decomposition,
        [8, 8, 8],
    )?;
    execute_phases(
        "density",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: rayon::current_num_threads()
                .min(assembly.decomposition.n_tasks())
                .max(1),
            ..Hints::default()
        },
        &env,
        &[],
        &assembly.work(),
    )?;
    Ok(env.output())
}
