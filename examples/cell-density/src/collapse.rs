// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Per-region counts: the cell table reduced to one row per
//! `(region, hemisphere)`, with the region's name from the tree and its
//! volume and cell density from the annotation.
//!
//! The grouping is `ops::rows`' three fragment phases (rows in, a partial per
//! block, a merge), so the counts are integer tallies and the mean size a
//! fixed-point sum: the table does not depend on how the rows' lattice is cut.

use std::collections::{BTreeMap, BTreeSet};

use atlas_regions::Ontology;
use blockflow::assemble::{Assembly, PlanBuilder};
use blockflow::env::ArrayEnvironment;
use blockflow::error::{Error, Result};
use blockflow::geometry::BlockGrid;
use blockflow::ops::rows::{
    collect_groups, Aggregate, GroupRowsOp, Grouping, MergeGroupsOp, Reduction, RowSourceOp,
    RowStreams, RowValues,
};
use blockflow::ops::tabulate::{from_signed_column, signed_column, FixedPoint};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::table::{Column, Schema, Value};
use blockflow::voxels::Voxels;
use blockflow::Dtype;

const CELLS: &str = "collapse.cells";
const PARTIALS: &str = "collapse.partials";
const GROUPS: &str = "collapse.groups";

/// One cell as the collapse sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionCell {
    /// Where the row sits on the grouping's lattice (any lattice holding every
    /// row; the atlas grid is the natural one).
    pub at: [usize; 3],
    pub region: i64,
    pub hemisphere: i64,
    pub size: f64,
}

/// One output row.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionRow {
    pub region: i64,
    pub hemisphere: i64,
    pub name: Option<String>,
    pub acronym: Option<String>,
    pub cells: u64,
    /// `None` for a region with no cells.
    pub mean_size: Option<f64>,
    /// The region's annotation voxels on this hemisphere.
    pub volume_voxels: Option<u64>,
    pub volume_mm3: Option<f64>,
    /// Cells per cubic millimetre, where the volume is known and non-zero.
    pub density_per_mm3: Option<f64>,
}

/// What the region table is built from besides the cells.
#[derive(Debug, Clone, Copy, Default)]
pub struct RegionContext<'a> {
    /// Names, and with [`Self::all_regions`] the full region list.
    pub ontology: Option<&'a Ontology>,
    /// `(region, hemisphere)` volumes, from `AtlasFrame::region_volumes`.
    pub volumes: Option<&'a BTreeMap<(i64, i64), f64>>,
    /// `(region, hemisphere)` voxel counts, from `AtlasFrame::region_voxels`.
    pub voxels: Option<&'a BTreeMap<(i64, i64), u64>>,
    /// List every region of the tree (on every hemisphere seen), with zero
    /// counts where there are no cells, rather than only the observed ones.
    pub all_regions: bool,
}

fn cell_schema() -> Schema {
    Schema::new(vec![
        Column::u64("region"),
        Column::u64("hemisphere"),
        Column::f64("size"),
    ])
    .expect("three distinct names")
}

fn grouping() -> Result<Grouping> {
    Grouping::new(
        cell_schema(),
        vec![0, 1],
        vec![Reduction::new(2, Aggregate::Sum)],
        FixedPoint::bits(0)?,
    )
}

/// The grouping plan over `cells`, on a lattice of `volume` cut into `block`.
pub fn plan(cells: &[RegionCell], volume: [usize; 3], block: [usize; 3]) -> Result<Assembly> {
    let mut encoded = Vec::with_capacity(cells.len());
    for cell in cells {
        if (0..3).any(|axis| cell.at[axis] >= volume[axis]) {
            return Err(Error::invalid(format!(
                "the cell at {:?} is outside the {volume:?} grouping lattice, so no block owns it",
                cell.at
            )));
        }
        if !cell.size.is_finite() || cell.size.fract() != 0.0 {
            return Err(Error::invalid(format!(
                "the cell at {:?} has size {}; sizes are voxel counts",
                cell.at, cell.size
            )));
        }
        encoded.push(RowValues::new(
            cell.at,
            vec![
                Value::U64(signed_column(cell.region as i128)?),
                Value::U64(signed_column(cell.hemisphere as i128)?),
                Value::F64(cell.size),
            ],
        ));
    }
    let grid = BlockGrid::new(volume, block)?;
    let lattice = grid.blocks_per_axis();
    let mut builder = PlanBuilder::new(volume, Dtype::Bool, grid);
    let source = builder.fragments(RowSourceOp::new(
        "collapse cells",
        CELLS,
        Lifecycle::DeleteOnExit,
        cell_schema(),
        encoded,
    ))?;
    let streams = RowStreams::new(
        CELLS,
        source.index(),
        PARTIALS,
        Lifecycle::DeleteOnExit,
        cell_schema(),
    )?;
    let grouped = builder.fragments(GroupRowsOp::new("collapse group", streams, grouping()?)?)?;
    builder.fragments(MergeGroupsOp::new(
        "collapse merge",
        PARTIALS,
        grouped.index(),
        lattice,
        grouping()?,
        GROUPS,
        Lifecycle::Persistent,
    )?)?;
    builder.finish()
}

/// Count cells per `(region, hemisphere)`.
///
/// Rows are ordered by region id, then hemisphere.
pub fn collapse(
    cells: &[RegionCell],
    volume: [usize; 3],
    block: [usize; 3],
    context: RegionContext<'_>,
) -> Result<Vec<RegionRow>> {
    let assembly = plan(cells, volume, block)?;
    let env = ArrayEnvironment::for_decomposition(
        Voxels::zeros(Dtype::Bool, volume)?,
        &assembly.decomposition,
        [8, 8, 8],
    )?;
    execute_phases(
        "collapse",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints::default(),
        &env,
        &[],
        &assembly.work(),
    )?;
    let grouping = grouping()?;
    let groups = collect_groups(
        &env,
        GROUPS,
        assembly.decomposition.phases.len() - 1,
        volume,
        &grouping,
    )?;
    let mut observed: BTreeMap<(i64, i64), (u64, f64)> = BTreeMap::new();
    for group in &groups {
        let key = (
            from_signed_column(group.key[0]),
            from_signed_column(group.key[1]),
        );
        observed.insert(key, (group.rows, group.sum(&grouping, 0)?));
    }

    let mut keys: BTreeSet<(i64, i64)> = observed.keys().copied().collect();
    if context.all_regions {
        let ontology = context.ontology.ok_or_else(|| {
            Error::invalid("listing every region needs the region tree (an ontology)")
        })?;
        let mut hemispheres: BTreeSet<i64> = observed.keys().map(|key| key.1).collect();
        if let Some(volumes) = context.volumes {
            hemispheres.extend(volumes.keys().map(|key| key.1));
        }
        if hemispheres.is_empty() {
            hemispheres.insert(-1);
        }
        for id in ontology.ids() {
            for hemisphere in &hemispheres {
                keys.insert((id, *hemisphere));
            }
        }
    }
    Ok(keys
        .into_iter()
        .map(|(region, hemisphere)| {
            let (cells, total) = observed
                .get(&(region, hemisphere))
                .copied()
                .unwrap_or((0, 0.0));
            let volume_mm3 = context
                .volumes
                .and_then(|volumes| volumes.get(&(region, hemisphere)).copied());
            RegionRow {
                region,
                hemisphere,
                name: context
                    .ontology
                    .and_then(|tree| tree.name(region))
                    .map(str::to_string),
                acronym: context
                    .ontology
                    .and_then(|tree| tree.acronym(region))
                    .map(str::to_string),
                cells,
                mean_size: (cells > 0).then(|| total / cells as f64),
                volume_voxels: context
                    .voxels
                    .and_then(|voxels| voxels.get(&(region, hemisphere)).copied()),
                volume_mm3,
                density_per_mm3: volume_mm3
                    .filter(|volume| *volume > 0.0)
                    .map(|volume| cells as f64 / volume),
            }
        })
        .collect())
}
