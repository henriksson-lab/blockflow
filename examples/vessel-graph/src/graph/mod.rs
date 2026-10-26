// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Skeleton to annotated vessel graph.
//!
//! The stages, each a function over the previous stage's output:
//!
//! 1. [`build`] — one vertex per skeleton voxel, one edge per adjacent pair
//!    ([`raw`], through `blockflow::ops::{coordinates, adjacency}`);
//! 2. [`radii`] — per-vertex radius in the vessel mask by an outward walk
//!    ([`radius`], through `blockflow::ops::walk`), and [`signal::measure`]
//!    for artery/vein channel values;
//! 3. [`clean`] — clusters of adjacent branch points collapse to one vertex;
//! 4. [`reduce`] — degree-2 chains contract to edges with polylines;
//! 5. [`VesselGraph::assemble`] — the reduced structure with every raw-vertex
//!    column carried to vertices, edges and polyline points ([`columns`]);
//! 6. [`VesselGraph::largest_component`], then edge lengths along the
//!    polylines ([`VesselGraph::set_lengths`]: `length` in voxels,
//!    `length_units` in µm);
//! 7. atlas: [`VesselGraph::rescale_to_atlas`] (or
//!    [`VesselGraph::transform_coordinates`] with any [`PointMap`] — an
//!    [`Affine`] between lattices, or a registration's point transform as a
//!    closure), [`VesselGraph::scale_radii`], [`annotate`];
//! 8. [`classify`] — arteries and veins on the line graph;
//! 9. [`branch_density`] — vertices splatted through `blockflow::ops::voxelize`.
//!
//! [`vessel_graph`] runs 1–6 from in-memory volumes, and
//! [`vessel_graph_stored`] the same stages over stored volumes, planned block
//! by block ([`planned`]): the same graph, without holding a volume.
//! Everything after works on a [`VesselGraph`], which is written as NGFF
//! object tables ([`write_graph_tables`]) or JSON ([`VesselGraph::to_json`]).
//!
//! The graph itself is resident: it grows with the vessels' length, not with
//! the volume, so the volumes are what need a plan and the graph does not.

pub mod annotate;
pub mod branch;
pub mod classify;
pub mod clean;
pub mod cli;
pub mod columns;
pub mod component;
pub mod planned;
pub mod radius;
pub mod raw;
pub mod reduce;
pub mod signal;
pub(crate) mod store;
pub mod tables;
pub mod vessel;

use std::sync::Arc;

use blockflow::error::{Error, Result};
use blockflow::voxels::Voxels;

use crate::profile::GRAPH;

pub use annotate::annotate;
pub use branch::{branch_density, BranchParams, Selection};
pub use classify::{classify, ClassifyParams, ClassifyTrace};
pub use clean::{clean, CleanGraph, CleanParams};
pub use columns::{Aggregate, Column, EdgeFold, RawColumn};
pub use component::{largest_component, Component};
pub use radius::{carried_radii, radii, RadiusMeasurement, RadiusParams, RadiusSearch};
pub use raw::{build, build_blocked, BuildParams, Csr, RawGraph};
pub use reduce::{reduce, ReduceParams, ReducedGraph};
pub use tables::{
    write_graph_tables, write_graph_tables_within, GraphTables, TableGeometry, TableMemory,
};
pub use vessel::{Affine, Carried, Lattice, PointMap, VesselGraph};

/// The volumes a graph is measured from. Only the skeleton is required; all
/// share its shape.
#[derive(Debug, Clone, Copy)]
pub struct Inputs<'a> {
    /// The thinned vessel mask; any non-zero voxel is set.
    pub skeleton: &'a Voxels,
    /// The vessel mask the radius walk reads (background at or below
    /// [`RadiusParams::threshold`]).
    pub vessels: Option<&'a Voxels>,
    /// µm per voxel.
    pub spacing: [f64; 3],
    /// Artery-channel intensity, read as `artery_raw`.
    pub artery_intensity: Option<&'a Voxels>,
    /// Artery-channel mask (non-zero is artery), read as `artery_binary`.
    pub artery_mask: Option<&'a Voxels>,
    /// Vein-channel intensity, read as `vein_raw`.
    pub vein_intensity: Option<&'a Voxels>,
    /// Vein-channel mask, read as `vein_binary`.
    pub vein_mask: Option<&'a Voxels>,
}

impl<'a> Inputs<'a> {
    pub fn new(skeleton: &'a Voxels, spacing: [f64; 3]) -> Self {
        Self {
            skeleton,
            vessels: None,
            spacing,
            artery_intensity: None,
            artery_mask: None,
            vein_intensity: None,
            vein_mask: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphParams {
    pub build: BuildParams,
    pub radius: RadiusParams,
    pub clean: CleanParams,
    /// How many voxels beyond the vessel wall the artery intensity is read.
    pub artery_shift_vx: f64,
    /// Keep only the largest connected component.
    pub largest_component: bool,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            build: BuildParams::default(),
            radius: RadiusParams::default(),
            clean: CleanParams::default(),
            artery_shift_vx: 3.0,
            largest_component: true,
        }
    }
}

/// Stages 1–6: skeleton to vessel graph, with radii and channel signals
/// carried to every level.
pub fn vessel_graph(inputs: &Inputs<'_>, params: &GraphParams) -> Result<VesselGraph> {
    let shape = inputs.skeleton.shape();
    for (name, volume) in [
        ("vessels", inputs.vessels),
        ("artery_intensity", inputs.artery_intensity),
        ("artery_mask", inputs.artery_mask),
        ("vein_intensity", inputs.vein_intensity),
        ("vein_mask", inputs.vein_mask),
    ] {
        if let Some(volume) = volume {
            if volume.shape() != shape {
                return Err(Error::invalid(format!(
                    "the {name} volume is {:?} and the skeleton {shape:?}; every input is read \
                     at the skeleton's vertex coordinates.",
                    volume.shape()
                )));
            }
        }
    }
    GRAPH.stage("build raw graph");
    let raw = build(inputs.skeleton, &params.build)?;
    let points = raw.coordinates();
    let mut carried = Vec::new();

    GRAPH.stage("radii");
    let axial_um = match inputs.vessels {
        Some(vessels) => {
            let search = Arc::new(RadiusSearch::new(params.radius.clone())?);
            let stops = search.stops(vessels, points)?;
            carried.extend(carried_radii(Arc::clone(&search), &stops, inputs.spacing)?);
            stops
                .iter()
                .map(|stop| search.units_axial(*stop, inputs.spacing))
                .collect()
        }
        None => vec![[0.0; 3]; points.len()],
    };
    GRAPH.stage("signals");
    let at = |shift: f64| signal::search_radii(&axial_um, inputs.spacing, shift);
    if let Some(volume) = inputs.artery_intensity {
        let values = signal::measure(
            volume,
            points,
            &at(params.artery_shift_vx)?,
            signal::Reduce::Max,
        )?;
        carried.push(Carried::max("artery_raw", Column::scalar(values)));
    }
    if let Some(volume) = inputs.vein_intensity {
        let values = signal::measure(volume, points, &at(0.0)?, signal::Reduce::Max)?;
        carried.push(Carried::max("vein_raw", Column::scalar(values)));
    }
    for (name, mask) in [
        ("artery_binary", inputs.artery_mask),
        ("vein_binary", inputs.vein_mask),
    ] {
        if let Some(mask) = mask {
            let values = signal::measure(mask, points, &at(0.0)?, signal::Reduce::Max)?;
            let flags: Vec<bool> = values.iter().map(|value| *value != 0.0).collect();
            carried.push(Carried::binary(name, Column::flags(&flags)));
        }
    }

    finish(raw, carried, params, inputs.spacing)
}

/// Stages 4–6 from a raw graph and the columns carried on it: clean, reduce,
/// assemble, the largest component (if asked) and the edge lengths. Each
/// structure is let go as soon as the next stage no longer needs it; the
/// graph keeps the raw coordinates and columns its carried columns read.
pub fn finish(
    mut raw: RawGraph,
    carried: Vec<Carried>,
    params: &GraphParams,
    spacing: [f64; 3],
) -> Result<VesselGraph> {
    GRAPH.stage("clean");
    let mut cleaned = clean(&raw, &params.clean)?;
    raw.drop_edges();
    // The polylines are what every edge column is folded over.
    GRAPH.stage("reduce");
    let reduced = reduce(&cleaned, &ReduceParams::default())?;
    cleaned.drop_edges();
    GRAPH.stage("assemble");
    let graph = VesselGraph::assemble(&raw, &cleaned, &reduced, &carried)?;
    drop((raw, cleaned, reduced, carried));
    GRAPH.stage("largest component, lengths");
    let mut graph = match params.largest_component {
        true => graph.largest_component()?,
        false => graph,
    };
    graph.set_lengths(spacing)?;
    Ok(graph)
}

/// Stages 1–6 over stored volumes, planned over blocks
/// ([`planned::measure_stored`]); the same graph [`vessel_graph`] gives.
pub fn vessel_graph_stored(
    inputs: &planned::StoredInputs,
    params: &GraphParams,
    plan: &planned::PlanParams,
) -> Result<VesselGraph> {
    GRAPH.stage("build and measure (planned)");
    let (raw, carried, lease) = planned::measure_stored(inputs, params, &params.build, plan)?;
    if crate::profile::enabled() {
        eprintln!(
            "profile graph: raw graph of {} vertices and {} edges",
            raw.n_vertices(),
            raw.n_edges()
        );
    }
    let mut graph = finish(raw, carried, params, inputs.spacing)?;
    if let Some(lease) = lease {
        graph.hold(lease);
    }
    Ok(graph)
}
