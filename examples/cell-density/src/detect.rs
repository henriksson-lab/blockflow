// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The cell detector as one planned run: a volume in, one row per cell out.
//!
//! | step | blockflow |
//! |---|---|
//! | background removal: the volume minus its per-plane opening, clamped at 0 | `ops::background::background_estimate`'s opening by [`crate::extreme`], `DifferenceCombine` |
//! | optional band-pass | `ops::interest::BlobResponseOp` (difference of Gaussians), clamped at 0 |
//! | prominence | `ops::reconstruct::HExtremaOp` (skipped at `h = 0`) |
//! | maxima | a window maximum ([`crate::extreme`]) or `ops::regional` |
//! | foreground gate | `VoxelwiseMapOp::threshold`, `LogicCombine` |
//! | one seed per maximum | `ops::detect`'s labelling and seam merge (26-connected), in [`SeedRowsOp`], which also numbers them |
//! | seed image | `classical_centers::CenterSeedsOp` |
//! | basins | `ops::tiled_watershed::TiledWatershedOp` (default) or `ops::watershed::ScalableWatershedOp` over the negated filtered volume, inside the foreground, from the global seeds |
//! | edge basins dropped | [`DropEdgeBasinsOp`] (tiled only) |
//! | `size`, `source`, `background` | `ops::tabulate`, twice (raw input, background-removed volume), each merge folded once |
//!
//! Everything is in the volume's own `[z, y, x]` order. The clamp after the
//! background removal matters: without it the voxels where the opening exceeds
//! the original (near faces) go negative, and the flat zero background is what
//! the foreground threshold is measured from.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use blockflow::assemble::{Assembly, ImageId, PlanBuilder};
use blockflow::classical_centers::CenterSeedsOp;
use blockflow::env::{ArrayEnvironment, Environment};
use blockflow::error::{Error, Result};
use blockflow::fragment::{
    BlockOutput, BlockView, Coverage, FragmentInput, FragmentOp, FragmentOutput, PhaseView,
    SeamFold, SidecarSize,
};
use blockflow::geometry::BlockGrid;
use blockflow::iterate::SubstageLimit;
use blockflow::log::Stats;
use blockflow::op::Chain;
use blockflow::op::{Anchor, BlockOp};
use blockflow::ops::background::{background_estimate, DifferenceCombine};
use blockflow::ops::components::Connectivity;
use blockflow::ops::detect::{
    centroid_points, merge_moments_with, owner_of, LabelRegionsOp, RegionMoments,
};
use blockflow::ops::element::{ElementShape, StructuringElement};
use blockflow::ops::interest::BlobResponseOp;
use blockflow::ops::label::WidenLabelsOp;
use blockflow::ops::local::{Narrowing, Rounding};
use blockflow::ops::reconstruct::{flooding_bound, HExtremaOp};
use blockflow::ops::regional;
use blockflow::ops::ridge::Boundary;
use blockflow::ops::rows::collect_rows;
use blockflow::ops::tabulate::{
    collect_tabulation, FixedPoint, MergeTabulationOp, TabulateValuesOp,
};
use blockflow::ops::tiled_watershed::{TiledWatershedConfig, TiledWatershedOp};
use blockflow::ops::voxelwise::{
    Logic, LogicCombine, NarrowOp, VoxelwiseMapOp, VoxelwiseMaskOp, WidenOp,
};
use blockflow::ops::watershed::{ScalableWatershedConfig, ScalableWatershedOp};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::table::{Column, RowBuilder, Schema, Value};
use blockflow::voxels::Voxels;
use blockflow::{Dtype, EventListener};

use crate::extreme::{extreme_filter, End};
use crate::params::{CellParams, Maxima, Watershed};

const PLATEAUX: &str = "cells.plateaux";
const MOMENTS: &str = "cells.moments";
/// The numbered seeds: one row per maximum, `label_id` and the plateau size.
pub const SEEDS: &str = "cells.seeds";
const SOURCE_PARTIALS: &str = "cells.source.partials";
const SOURCE_ROWS: &str = "cells.source";
const BACKGROUND_PARTIALS: &str = "cells.background.partials";
const BACKGROUND_ROWS: &str = "cells.background";

const SEEDS_MAGIC: u64 = 0x4345_4c4c_5345_4431;

/// One detected cell.
#[derive(Debug, Clone, PartialEq)]
pub struct CellRow {
    /// The basin's label: the seed's rank in the canonical (`[z, y, x]`
    /// lexicographic) order of all seeds, from 1. It does not depend on the
    /// block grid, and it is the value `labels/<name>` holds for this cell.
    pub label_id: u64,
    /// The seed voxel (the rounded centroid of the maximum it came from),
    /// `[z, y, x]` in the processed volume.
    pub at: [usize; 3],
    /// The basin's voxel count.
    pub size: u64,
    /// The maximum of the raw input over the basin.
    pub source: f64,
    /// The maximum of the background-removed volume over the basin.
    pub background: f64,
    /// Whether the row passes the size and intensity bands (and the crust,
    /// once [`crate::atlas::annotate_cells`] has run).
    pub passes_filter: bool,
    /// Atlas columns, when an annotation was given.
    pub atlas: Option<CellAtlas>,
}

/// The atlas columns of one cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellAtlas {
    pub region: i64,
    pub hemisphere: i64,
    /// Distance to the brain surface: in micrometres for an annotation in the
    /// sample's frame (`NaN` outside it), in atlas voxels when [`Self::atlas_at`]
    /// is set (read at the nearest edge voxel outside the atlas).
    pub distance_to_surface: f64,
    /// With a registration: the cell's position in atlas level-0 voxels,
    /// `[z, y, x]` (continuous, voxel centres at integers). `None` when the
    /// annotation was given in the sample's frame.
    pub atlas_at: Option<[f64; 3]>,
}

/// What one run produced.
#[derive(Debug)]
pub struct CellRun {
    /// Every basin that exists, in `label_id` order.
    pub rows: Vec<CellRow>,
    /// Seeds before empty basins were dropped.
    pub seeds: usize,
    /// The basin volume (`label_id` per voxel), when asked for.
    pub basins: Option<Voxels>,
}

impl CellRun {
    pub fn filtered(&self) -> impl Iterator<Item = &CellRow> {
        self.rows.iter().filter(|row| row.passes_filter)
    }
}

/// Choices about the plan that do not change a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOptions {
    /// Also widen the basins to `uint64` for publication as a label image.
    pub publish_labels: bool,
    /// The h-maxima's runaway guard; `None` is [`substage_limit_for`].
    pub substage_limit: Option<SubstageLimit>,
    /// With [`Watershed::Whole`] only: the largest foreground component
    /// (bounding box, voxels) flooded by one exact queue; larger ones fall
    /// back to block-local queues, which changes their basins. Cell basins
    /// are small, so nearly every component takes the exact path and the
    /// fallback only bounds memory for large bright artefacts.
    pub exact_component_max_bbox_voxels: usize,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            publish_labels: false,
            substage_limit: None,
            exact_component_max_bbox_voxels: ScalableWatershedConfig::default()
                .exact_component_max_bbox_voxels,
        }
    }
}

/// The h-maxima flood's neighbourhood: the six face neighbours, the same graph
/// `ops::regional` takes maxima over, so that `RMAX(HMAX_h(f))` is the extended
/// maxima transform and not a mixture of two neighbourhoods.
fn flood_element() -> StructuringElement {
    StructuringElement::from_radius(ElementShape::Ellipsoid, [1, 1, 1])
}

/// The h-maxima runaway guard: four times the geometric bound.
///
/// `ops::reconstruct::flooding_bound` is the length of a shortest path, and a
/// flood across a clamped (exactly flat) background walks one voxel per
/// substage around obstacles, so correct runs exceed it: on a 40 x 40 x 28
/// noisy fixture with a band-pass the flood took 114 substages against a bound
/// of 107. Four times is still two orders of magnitude below the voxel count.
pub fn substage_limit_for(volume: [usize; 3]) -> SubstageLimit {
    let geometric = flooding_bound(volume, &flood_element());
    SubstageLimit::of(geometric.substages().saturating_mul(4))
        .expect("four times a positive bound is positive")
}

/// The plan, and the coordinates its answer is read back from.
pub struct CellPlan {
    pub assembly: Assembly,
    pub volume: [usize; 3],
    /// The background-removed volume.
    pub background_image: ImageId,
    /// The volume the maxima are taken on and the basins flood: the band-pass
    /// when there is one, the background-removed volume otherwise.
    pub filtered_image: ImageId,
    /// The gated maxima mask, `1.0` / `0.0`.
    pub maxima_image: ImageId,
    /// The basins, `uint32`.
    pub labels_image: ImageId,
    /// The basins widened to `uint64`, when [`PlanOptions::publish_labels`].
    pub published_image: Option<ImageId>,
    seeds_phase: usize,
    source_phase: usize,
    background_phase: usize,
    fixed: FixedPoint,
}

impl CellPlan {
    /// Build the plan for a `volume` of `dtype` cut into `block`s.
    pub fn assemble(
        params: &CellParams,
        dtype: Dtype,
        volume: [usize; 3],
        block: [usize; 3],
        options: PlanOptions,
    ) -> Result<Self> {
        params.validate()?;
        let grid = BlockGrid::new(volume, block)?;
        let mut plan = PlanBuilder::new(volume, dtype, grid.clone());

        // In the default configuration the background-removed volume is the
        // input's own unsigned integers (the input minus a selection of it,
        // clamped at zero), so it is stored in that type and widened where it
        // is read: a quarter of the bytes of `f64` for a `uint16` volume,
        // through every phase that reads it. The same voxels.
        let store = match dtype {
            Dtype::U8 | Dtype::U16 | Dtype::U32
                if params.dog.is_none()
                    && params.h <= 0.0
                    && matches!(params.maxima, Maxima::Window { .. }) =>
            {
                dtype
            }
            _ => Dtype::F64,
        };
        let background = background_chain(&params.background_element()?)?;
        let background = plan.pixels(if store == Dtype::F64 {
            background
        } else {
            Chain::sequence(vec![
                background,
                Chain::op(NarrowOp::new(
                    "background store",
                    Narrowing::new(store, Rounding::TowardZero)?,
                )),
            ])
        })?;
        let filtered = match &params.dog {
            Some(scales) => plan.pixels(Chain::sequence(vec![
                Chain::op(BlobResponseOp::difference_of_gaussians(
                    "band-pass",
                    scales.sigma,
                    scales.sigma2,
                    scales.truncate,
                    Boundary::Reflect,
                )?),
                Chain::op(VoxelwiseMapOp::new("band-pass rectify", |value: f64| {
                    value.max(0.0)
                })),
            ]))?,
            None => background,
        };
        let filtered_image = filtered.image()?;
        // The filtered volume as `f64`, from wherever it is stored.
        let read_filtered = || {
            if store == Dtype::F64 {
                Chain::source(filtered_image, Dtype::F64)
            } else {
                Chain::sequence(vec![
                    Chain::source(filtered_image, store),
                    Chain::op(WidenOp::new("filtered widen")),
                ])
            }
        };
        // At `h = 0` the transform is the identity; skipping it saves a
        // materialised volume and a fixed-point loop.
        let mut maxima_input = filtered_image;
        if params.h > 0.0 {
            maxima_input = plan
                .iterate(HExtremaOp::maxima(
                    "h-maxima",
                    flood_element(),
                    params.h,
                    options
                        .substage_limit
                        .unwrap_or_else(|| substage_limit_for(volume)),
                )?)?
                .image()?;
        }
        let found = match params.maxima {
            Maxima::Window { radius } => {
                let window = StructuringElement::from_radius(ElementShape::Box, radius);
                // `v - max_W(v) >= 0` is exactly `max_W(v) == v`: the window
                // maximum is never below `v`, and `x - x` is `+0` for finite `x`.
                Chain::sequence(vec![
                    Chain::parallel(
                        vec![
                            Chain::op(VoxelwiseMapOp::identity("maxima value")),
                            extreme_filter("maxima window", &window, End::Highest),
                        ],
                        Box::new(DifferenceCombine::new("maxima shortfall")),
                    )?,
                    Chain::op(VoxelwiseMapOp::at_or_above("maxima is peak", 0.0, 1.0, 0.0)),
                ])
            }
            Maxima::Regional => {
                regional::append_to(&mut plan, PLATEAUX, Lifecycle::DeleteOnExit, Dtype::F64)?;
                maxima_input = ImageId::from(plan.n_phases());
                Chain::op(VoxelwiseMapOp::identity("maxima mask"))
            }
        };
        // The same strict comparison the watershed mask uses below, so a seed
        // always lies inside the region its basin may grow in.
        let gate = VoxelwiseMapOp::threshold("foreground gate", params.threshold, 1.0, 0.0);
        let gate = if maxima_input == filtered_image {
            Chain::op(gate)
        } else {
            Chain::sequence(vec![
                Chain::source(filtered_image, Dtype::F64),
                Chain::op(gate),
            ])
        };
        // Stored as `bool`, an eighth of the `f64` mask's bytes.
        let mut steps = Vec::new();
        if store != Dtype::F64 {
            steps.push(Chain::op(WidenOp::new("maxima widen")));
        }
        steps.push(Chain::parallel(
            vec![found, gate],
            Box::new(LogicCombine::new("maxima kept", Logic::And)),
        )?);
        steps.push(Chain::op(NarrowOp::to_mask("maxima store")));
        let maxima = plan.pixels(Chain::sequence(steps))?;

        let connectivity = Connectivity::FacesEdgesAndCorners;
        let moments = plan.fragments(
            LabelRegionsOp::new("region labelling", MOMENTS, Lifecycle::DeleteOnExit)
                .connecting(connectivity),
        )?;
        let seeds = plan.fragments(SeedRowsOp::new(
            moments.index(),
            connectivity,
            Lifecycle::Persistent,
        ))?;
        let seed_image = plan
            .fragments(CenterSeedsOp::new(
                "cell seeds",
                SEEDS,
                seeds.index(),
                seed_schema(),
            ))?
            .image()?;
        let mask = plan
            .pixels(Chain::sequence(vec![
                read_filtered(),
                Chain::op(VoxelwiseMaskOp::threshold("foreground", params.threshold)),
            ]))?
            .image()?;

        // The flood climbs a cost, and basins are wanted around bright
        // structure.
        let negate = || Chain::op(VoxelwiseMapOp::new("negated", |value: f64| -value));
        let labels = match params.watershed {
            Watershed::Whole => {
                // Negated on the block grid, where it runs in parallel. The
                // watershed answers from one global queue order per foreground
                // component, so it is planned as one block over the volume.
                plan.pixels(Chain::sequence(vec![read_filtered(), negate()]))?;
                plan.regrid(BlockGrid::new(volume, volume)?);
                plan.pixels(Chain::op(
                    ScalableWatershedOp::new(
                        "cell basins",
                        seed_image,
                        ScalableWatershedConfig {
                            exact_component_max_bbox_voxels: options
                                .exact_component_max_bbox_voxels,
                            block,
                            separation: params.separation,
                        },
                    )
                    .within(mask),
                ))?
                .image()?
            }
            Watershed::Tiled { tile, halo } => {
                // One phase on the tile grid: negate the buffer, flood it from
                // the global seeds in reach, and clear what edge seeds took, so
                // that every label left is a seed's `label_id`. Nothing but
                // the basins is materialised.
                let config = tiled_config(volume, tile, halo, params.separation)?;
                plan.regrid(BlockGrid::new(volume, config.tile)?);
                let edge_seed_base = config.edge_seed_base;
                plan.pixels(Chain::sequence(vec![
                    read_filtered(),
                    negate(),
                    Chain::op(
                        TiledWatershedOp::new("cell basins", config)?
                            .with_seeds(seed_image)
                            .within(mask),
                    ),
                    Chain::op(DropEdgeBasinsOp { edge_seed_base }),
                ]))?
                .image()?
            }
        };
        plan.regrid(grid.clone());
        let published_image = if options.publish_labels {
            // Straight after the watershed, whose image is this phase's input;
            // `uint64` is what the label pyramid reads.
            Some(plan.pixels(Chain::op(WidenLabelsOp))?.image()?)
        } else {
            None
        };

        // Only `count` and `max` are read: a tally and a selection, both exact
        // at any fixed point, so the coarsest one has the most range.
        let fixed = FixedPoint::bits(0)?;
        let lattice = grid.blocks_per_axis();
        let mut tabulate = |values: ImageId,
                            partials: &'static str,
                            rows: &'static str,
                            names: [&'static str; 2]|
         -> Result<usize> {
            let partial = plan.fragments(TabulateValuesOp::new(
                names[0],
                labels,
                values,
                fixed,
                partials,
                Lifecycle::DeleteOnExit,
            )?)?;
            Ok(plan
                .fragments(
                    MergeTabulationOp::new(
                        names[1],
                        partials,
                        partial.index(),
                        lattice,
                        fixed,
                        rows,
                        Lifecycle::Persistent,
                    )
                    .folded_once(),
                )?
                .index())
        };
        let source_phase = tabulate(
            ImageId::from(0usize),
            SOURCE_PARTIALS,
            SOURCE_ROWS,
            ["source tabulation", "source merge"],
        )?;
        let background_image = background.image()?;
        let background_phase = tabulate(
            background_image,
            BACKGROUND_PARTIALS,
            BACKGROUND_ROWS,
            ["background tabulation", "background merge"],
        )?;

        Ok(Self {
            volume,
            background_image,
            filtered_image,
            maxima_image: maxima.image()?,
            labels_image: labels,
            published_image,
            seeds_phase: seeds.index(),
            source_phase,
            background_phase,
            fixed,
            assembly: plan.finish()?,
        })
    }

    /// Hints at the width of the machine (rayon's thread count unless
    /// `workers` says otherwise), keeping `keep` alive past its last reader.
    pub fn hints(&self, workers: Option<usize>, keep: &[ImageId]) -> Hints {
        let workers = workers.unwrap_or_else(rayon::current_num_threads).max(1);
        Hints {
            concurrency: workers.min(self.assembly.decomposition.n_tasks()).max(1),
            keep_images: keep.iter().copied().collect(),
            ..Hints::default()
        }
    }

    /// Run the plan in `env`.
    pub fn execute(
        &self,
        name: &'static str,
        env: &dyn Environment,
        hints: &Hints,
        listeners: &[Arc<dyn EventListener>],
    ) -> Result<Stats> {
        execute_phases(
            name,
            &self.assembly.workflow,
            &self.assembly.decomposition,
            hints,
            env,
            listeners,
            &self.assembly.work(),
        )
    }

    /// The numbered seeds a run left behind: `(label_id, at)`, in label order.
    pub fn seeds(&self, env: &dyn Environment) -> Result<Vec<(u64, [usize; 3])>> {
        collect_rows(env, SEEDS, self.seeds_phase, self.volume, seed_schema())?
            .into_iter()
            .map(|row| match row.values.first() {
                Some(Value::U64(label)) => Ok((*label, row.at)),
                other => Err(Error::invalid(format!(
                    "a seed row carries {other:?} where its label should be"
                ))),
            })
            .collect()
    }

    /// The cells a run left behind, flagged by `params`' bands.
    pub fn cells(&self, env: &dyn Environment, params: &CellParams) -> Result<CellRun> {
        let seeds = self.seeds(env)?;
        let source: BTreeMap<u64, (u64, f64)> =
            collect_tabulation(env, SOURCE_ROWS, self.source_phase, self.volume, self.fixed)?
                .into_iter()
                .map(|row| (row.label, (row.count, row.max)))
                .collect();
        let background: BTreeMap<u64, f64> = collect_tabulation(
            env,
            BACKGROUND_ROWS,
            self.background_phase,
            self.volume,
            self.fixed,
        )?
        .into_iter()
        .map(|row| (row.label, row.max))
        .collect();
        let known: BTreeSet<u64> = seeds.iter().map(|(label, _)| *label).collect();
        if let Some(stray) = source.keys().find(|label| !known.contains(label)) {
            return Err(Error::invalid(format!(
                "the basin volume holds label {stray}, which no seed carries"
            )));
        }
        let mut rows = Vec::with_capacity(source.len());
        for (label_id, at) in &seeds {
            // A seed whose basin is empty (the line can clear it) is not a cell.
            let Some(&(size, source)) = source.get(label_id) else {
                continue;
            };
            let background = background.get(label_id).copied().unwrap_or(f64::NAN);
            rows.push(CellRow {
                label_id: *label_id,
                at: *at,
                size,
                source,
                background,
                passes_filter: params.passes(size as f64, source),
                atlas: None,
            });
        }
        rows.sort_by_key(|row| row.label_id);
        Ok(CellRun {
            rows,
            seeds: seeds.len(),
            basins: None,
        })
    }
}

/// The tiled watershed's configuration for `volume`: the tile and halo clamped
/// to the volume (one tile on an axis the tile covers, where the halo has
/// nothing to add), and the edge-seed id space shared out over the tiles.
///
/// The clamping changes no voxel: on an axis with one tile the buffer is the
/// whole axis whatever the halo.
pub fn tiled_config(
    volume: [usize; 3],
    tile: [usize; 3],
    halo: [usize; 3],
    separation: blockflow::ops::watershed::Separation,
) -> Result<TiledWatershedConfig> {
    let tile: [usize; 3] = std::array::from_fn(|a| tile[a].min(volume[a]).max(1));
    let halo: [usize; 3] = std::array::from_fn(|a| halo[a].min(tile[a] - 1));
    let config = TiledWatershedConfig::new(tile, halo)?.with_separation(separation);
    // Edge seeds are numbered `edge_seed_base + tile * ids_per_tile + k`, and
    // the lot must stay in `u32`. They are cleared straight away, so only
    // their count per tile matters; a tile with more is refused by name.
    let tiles: u64 = (0..3).map(|a| volume[a].div_ceil(tile[a]) as u64).product();
    let room = (u64::from(u32::MAX) - u64::from(config.edge_seed_base)) / tiles.max(1);
    let ids_per_tile = room.min(1 << 16) as u32;
    if ids_per_tile < 256 {
        return Err(Error::invalid(format!(
            "{tiles} watershed tiles of {tile:?} leave {ids_per_tile} edge-seed ids per tile; \
             use larger tiles"
        )));
    }
    Ok(config.with_ids_per_tile(ids_per_tile))
}

/// Clears every voxel the tiled watershed gave to an edge seed (an id at or
/// above `edge_seed_base`).
///
/// An edge seed marks the part of a basin whose global seed lies beyond the
/// halo. Keeping it would add a row no seed carries; giving it to a
/// neighbour is the leak the edge seed exists to prevent. So the part is
/// dropped, and that cell's size is short by it. With a halo of a few cell
/// diameters this is rare (see the README for the count on the real crop).
#[derive(Debug, Clone, Copy)]
pub struct DropEdgeBasinsOp {
    pub edge_seed_base: u32,
}

impl BlockOp for DropEdgeBasinsOp {
    fn name(&self) -> &'static str {
        "drop edge basins"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::U32
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U32
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        let base = self.edge_seed_base;
        out.view_mut::<u32>()?
            .zip_mut_with(&input.view::<u32>()?, |o, &v| {
                *o = if v >= base { 0 } else { v }
            });
        Ok(())
    }
}

/// The background removal: the original minus its opening, clamped at zero,
/// with the opening at the input's own width.
///
/// The opening is a rank filter and selects an existing value, so it is exact
/// in `uint16`, and it is most of the detector's CPU; `WidenOp` in both arms is
/// what lets the input stay narrow (the identity `VoxelwiseMapOp` accepts only
/// `f64`). Everything after the difference is `f64`.
fn background_chain(element: &StructuringElement) -> Result<Chain> {
    Ok(Chain::sequence(vec![
        Chain::parallel(
            vec![
                Chain::op(WidenOp::new("background original")),
                Chain::sequence(vec![
                    opening(element),
                    Chain::op(WidenOp::new("background estimate")),
                ]),
            ],
            Box::new(DifferenceCombine::new("background difference")),
        )?,
        Chain::op(VoxelwiseMapOp::new("background rectify", |value: f64| {
            value.max(0.0)
        })),
    ]))
}

/// `ops::background::background_estimate`'s opening (the erosion, then the
/// dilation by the reflected element), with each pass by running extremes over
/// the whole buffer; see [`crate::extreme`]. The same voxels.
fn opening(element: &StructuringElement) -> Chain {
    match element.reflected() {
        Ok(reflected) => Chain::sequence(vec![
            extreme_filter("background.lowest", element, End::Lowest),
            extreme_filter("background.highest", &reflected, End::Highest),
        ]),
        Err(_) => background_estimate(element),
    }
}

/// The seed rows' schema: the label, and the size of the maximum's plateau.
pub fn seed_schema() -> Schema {
    Schema::new(vec![Column::u64("label_id"), Column::f64("plateau_voxels")])
        .expect("two distinct names")
}

/// Merges the maxima's components across blocks, and numbers their seeds
/// `1..=n` in the canonical order of their positions.
///
/// A barrier, and the one place the components are merged: its `reduce` runs
/// `ops::detect`'s seam merge over every block's moments once, takes each
/// component's point (the rounded centroid, its voxel count as the weight),
/// keeps one point per position (the lowest weight, as a block's own sorted
/// list would), numbers them in position order and groups them by the block
/// that owns the position. Each block then copies out its own rows. The label
/// is a function of the point set alone, so it does not move with the grid,
/// and it fits the `uint32` the watershed's seed image holds up to 4.29e9
/// seeds.
///
/// This is `ops::detect::RegionPointsOp` and a numbering in one phase, without
/// the per-block step that op takes: there every block decodes every component
/// in the volume to find its own, which on a whole tile (1 088 blocks, a
/// million components) cost about 480 s of CPU and 4 GB per worker's copy.
struct SeedRowsOp {
    moments_phase: usize,
    connectivity: Connectivity,
    lifecycle: Lifecycle,
    schema: Schema,
}

impl SeedRowsOp {
    fn new(moments_phase: usize, connectivity: Connectivity, lifecycle: Lifecycle) -> Self {
        Self {
            moments_phase,
            connectivity,
            lifecycle,
            schema: seed_schema(),
        }
    }
}

/// Words a seed takes in the reduced blob: `z, y, x, label, weight bits`.
const SEED_WORDS: usize = 5;

fn word(bytes: &[u8], index: usize) -> Option<u64> {
    let at = index.checked_mul(8)?;
    let bytes = bytes.get(at..at.checked_add(8)?)?;
    Some(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
}

impl FragmentOp for SeedRowsOp {
    fn name(&self) -> &'static str {
        "cell seed numbering"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        false
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn barrier(&self) -> bool {
        true
    }

    fn inputs(&self) -> Vec<FragmentInput> {
        vec![FragmentInput::own(MOMENTS, self.moments_phase).with_reach([0, 0, 0])]
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(SEEDS, self.lifecycle, Coverage::EveryBlock)
                .sized(SidecarSize::row_table(&self.schema, 1)),
        ]
    }

    fn gathers(&self) -> bool {
        false
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::Unordered)
    }

    /// `[magic, blocks, offsets[blocks + 1], seeds...]` in words, the seeds
    /// grouped by owning block (row-major) and in position order within it;
    /// `offsets` count seeds.
    fn reduce(&self, at: &PhaseView<'_>) -> Result<Vec<u8>> {
        let mut reports = BTreeMap::new();
        for (key, bytes) in at.fragments(MOMENTS)? {
            reports.insert(key.block, RegionMoments::decode(&bytes)?);
        }
        let lattice = at.grid().blocks_per_axis();
        let components = merge_moments_with(&reports, lattice, self.connectivity)?;
        drop(reports);
        let mut points = centroid_points(&components);
        drop(components);
        points.dedup_by_key(|point| point.at);
        if points.len() > u32::MAX as usize {
            return Err(Error::invalid(format!(
                "{} seeds do not fit the watershed's uint32 seed image; cut the volume into \
                 slabs or raise the foreground threshold",
                points.len()
            )));
        }
        let linear = |index: [usize; 3]| (index[0] * lattice[1] + index[1]) * lattice[2] + index[2];
        let mut seeds: Vec<(usize, u64, &blockflow::points::Point)> = points
            .iter()
            .enumerate()
            .map(|(rank, point)| {
                (
                    linear(owner_of(at.grid(), point.at)),
                    rank as u64 + 1,
                    point,
                )
            })
            .collect();
        // Position order is label order, so (owner, label) is (owner, position).
        seeds.sort_unstable_by_key(|&(owner, label, _)| (owner, label));
        let blocks = lattice.iter().product::<usize>();
        let mut words = Vec::with_capacity(3 + blocks + seeds.len() * SEED_WORDS);
        words.push(SEEDS_MAGIC);
        words.push(blocks as u64);
        let mut next = 0usize;
        for block in 0..=blocks {
            while next < seeds.len() && seeds[next].0 < block {
                next += 1;
            }
            words.push(next as u64);
        }
        for (_, label, point) in &seeds {
            words.extend(point.at.map(|value| value as u64));
            words.push(*label);
            words.push(point.weight.to_bits());
        }
        let mut bytes = Vec::with_capacity(words.len() * 8);
        for value in words {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        let sorted = at.reduced;
        let lattice = at.grid.blocks_per_axis();
        let blocks = lattice.iter().product::<usize>();
        if word(sorted, 0) != Some(SEEDS_MAGIC) || word(sorted, 1) != Some(blocks as u64) {
            return Err(Error::invalid(
                "the seed numbering has no sorted seed list for this lattice",
            ));
        }
        let index = at.index;
        let linear = (index[0] * lattice[1] + index[1]) * lattice[2] + index[2];
        let (Some(start), Some(end), Some(total)) = (
            word(sorted, 2 + linear),
            word(sorted, 3 + linear),
            word(sorted, 2 + blocks),
        ) else {
            return Err(Error::invalid("the sorted seed list's offsets are short"));
        };
        let base = 3 + blocks;
        if start > end || end > total || sorted.len() != (base + total as usize * SEED_WORDS) * 8 {
            return Err(Error::invalid("the sorted seed list has the wrong length"));
        }
        let mut rows = RowBuilder::new(Arc::new(self.schema.clone()));
        for seed in start as usize..end as usize {
            let at = |w: usize| word(sorted, base + seed * SEED_WORDS + w).expect("checked length");
            rows.push(
                [at(0) as usize, at(1) as usize, at(2) as usize],
                &[Value::U64(at(3)), Value::F64(f64::from_bits(at(4)))],
            )?;
        }
        Ok(BlockOutput::fragment(SEEDS, rows.encode()))
    }
}

/// Run the detector over an in-memory volume.
///
/// Every intermediate image is held in memory, so this is for tests and for
/// volumes of a few hundred million voxels; the command line runs the same
/// plan over OME-Zarr.
pub fn detect_cells(
    input: Voxels,
    params: &CellParams,
    block: [usize; 3],
    keep_basins: bool,
) -> Result<CellRun> {
    let plan = CellPlan::assemble(
        params,
        input.dtype(),
        input.shape(),
        block,
        PlanOptions::default(),
    )?;
    let env = ArrayEnvironment::for_decomposition(input, &plan.assembly.decomposition, [8, 8, 8])?;
    let keep = if keep_basins {
        vec![plan.labels_image]
    } else {
        Vec::new()
    };
    plan.execute("cells", &env, &plan.hints(None, &keep), &[])?;
    let mut run = plan.cells(&env, params)?;
    if keep_basins {
        run.basins = Some(env.try_image(plan.labels_image.index())?);
    }
    Ok(run)
}
