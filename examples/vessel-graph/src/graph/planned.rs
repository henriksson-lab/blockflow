// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Stages 1–3 of the graph planned over blocks of stored volumes: the raw
//! graph, the radii and the channel signals, without holding a volume whole.
//!
//! Two plans, over the published labels and channels attached as stored
//! arrays:
//!
//! ```text
//! vessel mask, skeleton, artery / vein masks (u64) -> pack (one u8 image, a bit each)
//! packed masks + artery / vein channels -> vertex measure (fragments: rows and edges)
//! ```
//!
//! **The pack** ([`PackMasksOp`]) narrows the four `uint64` labels to one
//! byte a voxel, reading each label chunk once (no chunk cache), on blocks of
//! [`PlanParams::pack_block`].
//!
//! **The vertex measure** ([`VertexMeasureOp`]) is a fragment phase on blocks
//! of [`PlanParams::block`] that reads the packed masks and the channels over
//! the block's core grown by one reach: the radius walk's stated maximum
//! ([`super::RadiusParams::max_radius`]) or the largest search radius any
//! signal can be read over, whichever is larger. For each skeleton voxel of its
//! core it walks the vessel mask ([`super::RadiusSearch::stop_at`]) and reads
//! each channel over the vertex's own ball ([`super::signal::Ball`]), exactly
//! as the resident stages do over the whole volume: the walk and the balls
//! never leave the window inside the volume, so every vertex gets the answer
//! the whole volume gives, on every block grid. Its rows are the vertex list
//! itself (the skeleton's set voxels), merged in lexicographic order, and
//! each row says which of the voxel's lexicographically later neighbours
//! (`blockflow::ops::adjacency::forward_offsets`) are skeleton too: the
//! skeleton's adjacent pairs, a bit each, in the order
//! [`super::build_blocked`] lists them ([`RawGraph::from_steps`]).
//!
//! **The collect** keeps each column in its narrowest store as it decodes
//! the rows: a key and a row index per vertex for the sort, the stop as a
//! `u32` (the four radius arrays are read off it), the neighbour bits, an
//! intensity of a `uint16` channel as an `f32` and a mask as a byte.

use std::path::PathBuf;

use blockflow::assemble::{Assembly, ImageId, PlanBuilder};
use blockflow::budget::{process_budget, process_cap, Lease, MemoryCap};
use blockflow::decomposition::CostModel;
use blockflow::env::{BlockBuf, Environment};
use blockflow::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, SeamFold,
    SourceBlocks,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInput, SourceInputs};
use blockflow::ops::adjacency::forward_offsets;
use blockflow::reach::Reach;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{choose_grid, execute_phases, expected_bytes_per_block, Hints, Host};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::AttachedImage;
use blockflow::{Dtype, Error, Result, ZarrEnvironment};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

use crate::profile::{Listening, GRAPH};

use super::columns::RawColumn;
use super::radius::carried_stops;
use super::raw::{BuildParams, RawGraph};
use super::signal::{search_radius, shift_um, Ball, Reduce};
use super::store::{Narrow, Stops, Values};
use super::vessel::Carried;
use super::{GraphParams, RadiusSearch};

const MEASURE_STREAM: &str = "graph.vertex-measure";
/// Per block, how many vertex rows it wrote: the graph's size, known before
/// the rows are read.
const COUNT_STREAM: &str = "graph.vertex-count";
const MEASURE_MAGIC: &[u8] = b"vessel-graph vertex measure v2";
/// A row's fixed part: the coordinate, the stop and the neighbour bits.
const ROW_HEAD: usize = 12 + 4 + 2;

/// The stored volumes a planned graph reads, each level 0 of an OME-Zarr
/// array, all of one shape.
#[derive(Debug, Clone)]
pub struct StoredInputs {
    pub skeleton: AttachedImage,
    /// The vessel mask the radius walk reads.
    pub vessels: AttachedImage,
    pub artery_intensity: Option<AttachedImage>,
    pub artery_mask: Option<AttachedImage>,
    pub vein_intensity: Option<AttachedImage>,
    pub vein_mask: Option<AttachedImage>,
    /// µm per voxel.
    pub spacing: [f64; 3],
}

/// How the planned stages run. None of it changes a result.
#[derive(Debug, Clone)]
pub struct PlanParams {
    /// Block edge of the vertex measure, `z, y, x`; `None` for the grid the
    /// planner prices fastest under the memory cap
    /// (`blockflow::strategy::choose_grid`).
    pub block: Option<[usize; 3]>,
    /// Block edge of the pack, `z, y, x`: what holds the `uint64` labels;
    /// `None` for the planner's choice.
    pub pack_block: Option<[usize; 3]>,
    pub workers: usize,
    /// The chunk cache of the stored arrays, in bytes.
    pub cache_bytes: u64,
    /// Scratch directory for the packed masks; removed afterwards.
    pub work: PathBuf,
    /// What the resident graph holds per raw vertex at its widest, leased
    /// from the process's memory budget once the vertices are counted (and
    /// held by the graph); `0` leases nothing. See [`GRAPH_BYTES_PER_VERTEX`].
    pub resident_bytes_per_vertex: u64,
}

/// What the resident graph holds per raw vertex at its widest, measured on
/// the osf-sa3x8 region and crop (`tests/graph_memory.rs`, which also holds
/// a synthetic lattice to it): about 43 bytes while the vertex rows are
/// collected and sorted, 54 while the graph is cleaned and 62 while it is
/// reduced (the raw coordinates, 12, the radius stops, 4, and the signals,
/// 5, beside the cleaned edges and their incidence), 40 for the assembled
/// graph; with room for the allocator.
pub const GRAPH_BYTES_PER_VERTEX: u64 = 72;

/// What the atlas columns add per raw vertex (about a polyline point per raw
/// vertex): `coordinates_atlas` as three `f64`s, the region, hemisphere and
/// distance to surface each narrowed, and the lookups' batch of a quarter of
/// a million points; measured at about 37 on the osf-sa3x8 region.
pub const ATLAS_BYTES_PER_VERTEX: u64 = 48;

/// The pack's compute per voxel of each label it reads, in the planner's
/// unit (about a nanosecond): measured at 27 on the osf-sa3x8 region (7.3 s
/// of CPU for three `uint64` labels of 67 M voxels), almost all of it the
/// label chunks' decoding.
const PACK_COST_PER_IMAGE: f64 = 27.0;

/// The vertex measure's compute per voxel of its block: measured at 134 on
/// the osf-sa3x8 region (9.0 s of CPU over 67 M voxels: the masks' bits and,
/// per skeleton voxel, the radius walk and the signal balls), so it grows
/// with the skeleton's density.
const MEASURE_COST: f64 = 134.0;

const VESSEL: u8 = 1;
const SKELETON: u8 = 2;
const ARTERY: u8 = 4;
const VEIN: u8 = 8;

/// The published `uint64` masks packed into one `u8` image, a bit each: the
/// input (the vessel mask) as [`VESSEL`] and each source as its bit.
pub struct PackMasksOp {
    sources: Vec<(ImageId, Dtype, u8)>,
}

impl BlockOp for PackMasksOp {
    fn name(&self) -> &'static str {
        "pack masks"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn accepts(&self, _dtype: Dtype) -> bool {
        true
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::U8
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.sources
            .iter()
            .map(|(image, dtype, _)| SourceInput::voxelwise(*image).holding(*dtype))
            .collect()
    }

    fn apply(&self, _input: &Voxels, _out: &mut Voxels, _at: &Anchor) -> Result<()> {
        Err(Error::invalid(
            "pack masks reads its sources and is applied through `apply_with`",
        ))
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        _at: &Anchor,
    ) -> Result<()> {
        let mut out = out.view_mut::<u8>()?;
        let shape = input.shape();
        if [out.shape()[0], out.shape()[1], out.shape()[2]] != shape {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                got: out.shape().to_vec(),
            });
        }
        or_bit(&mut out, input, VESSEL, true)?;
        for (image, _, bit) in &self.sources {
            or_bit(&mut out, sources.get(*image)?, *bit, false)?;
        }
        Ok(())
    }

    fn cost_per_voxel(&self) -> f64 {
        PACK_COST_PER_IMAGE * (1.0 + self.sources.len() as f64)
    }
}

/// `out |= bit` where `mask` is set (`out = bit` where `first`).
fn or_bit(
    out: &mut ndarray::ArrayViewMut3<'_, u8>,
    mask: &Voxels,
    bit: u8,
    first: bool,
) -> Result<()> {
    if mask.shape() != [out.shape()[0], out.shape()[1], out.shape()[2]] {
        return Err(Error::ShapeMismatch {
            expected: out.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }
    let set = |slot: &mut u8, on: bool| {
        let value = if on { bit } else { 0 };
        *slot = if first { value } else { *slot | value };
    };
    match mask {
        Voxels::U64(array) => out.zip_mut_with(array, |slot, value| set(slot, *value != 0)),
        Voxels::Bool(array) => out.zip_mut_with(array, |slot, value| set(slot, *value)),
        other => out.zip_mut_with(&other.nonzero(), |slot, value| set(slot, *value)),
    }
    Ok(())
}

/// One channel read around every vertex: a stored channel, or a bit of the
/// packed masks.
#[derive(Debug, Clone, Copy)]
enum Source {
    Channel(ImageId, Dtype),
    Bit(u8),
}

#[derive(Debug, Clone, Copy)]
struct Signal {
    source: Source,
    /// The widening of the search radius beyond the vessel, in µm.
    shift_um: f64,
}

/// Per skeleton voxel of the core: the radius walk's stop, which later
/// neighbours are skeleton too, and each signal's maximum over the vertex's
/// ball. See the module header.
pub struct VertexMeasureOp {
    signals: Vec<Signal>,
    search: Arc<RadiusSearch>,
    ball: Arc<Ball>,
    spacing: [f64; 3],
    reach: usize,
    steps: Vec<[isize; 3]>,
}

impl VertexMeasureOp {
    fn row_bytes(&self) -> usize {
        ROW_HEAD + 8 * self.signals.len()
    }
}

impl FragmentOp for VertexMeasureOp {
    fn name(&self) -> &'static str {
        "vertex measure"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        self.reach
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![
            FragmentOutput::new(
                MEASURE_STREAM,
                Lifecycle::DeleteOnExit,
                Coverage::EveryBlock,
            ),
            FragmentOutput::new(COUNT_STREAM, Lifecycle::DeleteOnExit, Coverage::EveryBlock),
        ]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.signals
            .iter()
            .filter_map(|signal| match signal.source {
                Source::Channel(image, dtype) => {
                    Some(SourceInput::new(image, Reach::from([self.reach; 3])).holding(dtype))
                }
                Source::Bit(_) => None,
            })
            .collect()
    }

    fn cost_per_voxel(&self) -> f64 {
        MEASURE_COST
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        // Each row is a function of its own voxel and the windows around it,
        // each pair of its lower voxel's neighbourhood.
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, at: &BlockView<'_>) -> Result<BlockOutput> {
        self.apply_with(at, SourceBlocks::none())
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let BlockBuf::Array(packed) = at.pixels()? else {
            return Err(Error::invalid(
                "vertex measure: a run without data has no vertices to measure",
            ));
        };
        let read = at.read.shape3();
        if packed.shape() != read {
            return Err(Error::invalid(format!(
                "vertex measure: the packed masks arrived as {:?} for a read extent of {read:?}",
                packed.shape()
            )));
        }
        let packed = packed.view::<u8>()?;
        let origin = [at.read.start[0], at.read.start[1], at.read.start[2]];
        let volume = at.volume();
        let bit = |bit: u8| Voxels::Bool(packed.mapv(|value| value & bit != 0));
        let vessels = bit(VESSEL);
        let skeleton = packed.mapv(|value| value & SKELETON != 0);
        let mut owned = Vec::new();
        for signal in &self.signals {
            if let Source::Bit(mask) = signal.source {
                owned.push((mask, bit(mask)));
            }
        }
        let windows = self
            .signals
            .iter()
            .map(|signal| match signal.source {
                Source::Channel(image, _) => match sources.get(image.index())? {
                    BlockBuf::Array(array) if array.shape() == read => Ok(array),
                    _ => Err(Error::invalid(format!(
                        "vertex measure: image {} arrived without data or at another extent                          than {read:?}",
                        image.index()
                    ))),
                },
                Source::Bit(mask) => Ok(&owned
                    .iter()
                    .find(|(owned, _)| *owned == mask)
                    .expect("every bit signal has a window")
                    .1),
            })
            .collect::<Result<Vec<&Voxels>>>()?;

        // One row per skeleton voxel of the core, plane by plane in parallel,
        // concatenated in plane order.
        let core = at.core;
        let planes = (core.start[0]..core.start[0] + core.shape[0])
            .into_par_iter()
            .map(|z| -> Result<Vec<u8>> {
                let mut bytes = Vec::new();
                for y in core.start[1]..core.start[1] + core.shape[1] {
                    for x in core.start[2]..core.start[2] + core.shape[2] {
                        if !skeleton[[z - origin[0], y - origin[1], x - origin[2]]] {
                            continue;
                        }
                        let point = [z as u32, y as u32, x as u32];
                        let stopped = self.search.stop_at(&vessels, origin, volume, point)?;
                        let units = self.search.units_axial(stopped, self.spacing);
                        for value in point {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                        bytes.extend_from_slice(&Stops::encode(stopped)?.to_le_bytes());
                        let mut later = 0u16;
                        'steps: for (bit, step) in self.steps.iter().enumerate() {
                            let mut local = [0usize; 3];
                            for axis in 0..3 {
                                let moved = [z, y, x][axis] as isize + step[axis];
                                if moved < 0 || moved >= volume[axis] as isize {
                                    continue 'steps;
                                }
                                local[axis] = moved as usize - origin[axis];
                            }
                            if skeleton[local] {
                                later |= 1 << bit;
                            }
                        }
                        bytes.extend_from_slice(&later.to_le_bytes());
                        for (window, signal) in windows.iter().zip(&self.signals) {
                            let reach = search_radius(units, self.spacing, signal.shift_um);
                            let value = self.ball.reduce_at(
                                window,
                                origin,
                                volume,
                                point,
                                reach,
                                Reduce::Max,
                            )?;
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                    }
                }
                Ok(bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut rows = MEASURE_MAGIC.to_vec();
        for plane in planes {
            rows.extend_from_slice(&plane);
        }
        let count = ((rows.len() - MEASURE_MAGIC.len()) / self.row_bytes()) as u64;
        Ok(BlockOutput::fragment(MEASURE_STREAM, rows)
            .with_fragment(COUNT_STREAM, count.to_le_bytes().to_vec()))
    }
}

/// The rows of a finished run, in vertex (lexicographic) order.
struct Measured {
    coordinates: Vec<[u32; 3]>,
    /// [`Stops::encode`]d.
    stops: Vec<u32>,
    /// A bit per later neighbour that is skeleton too.
    later: Vec<u16>,
    /// One column per signal, each in the store its source's type fits.
    values: Vec<Values>,
}

/// How many vertex rows the blocks wrote.
fn count_measured(env: &dyn Environment) -> Result<usize> {
    let mut total = 0u64;
    fold_fragments(env, COUNT_STREAM, &mut |_key, fragment| {
        let count: [u8; 8] = fragment
            .try_into()
            .map_err(|_| Error::invalid("vertex measure: a count of the wrong shape"))?;
        total += u64::from_le_bytes(count);
        Ok(())
    })?;
    usize::try_from(total).map_err(|_| Error::invalid("vertex measure: too many vertices"))
}

/// The `rows` vertex rows of every block, decoded into compact columns and
/// put in vertex order. `stores` names each signal's store.
fn collect_measured(env: &dyn Environment, rows: usize, stores: &[Narrow]) -> Result<Measured> {
    let row_bytes = ROW_HEAD + 8 * stores.len();
    // Per row, its position packed into one sortable key and its index in
    // arrival order; the columns in arrival order. Sized once: a vector grown
    // by doubling can hold twice what it needs.
    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(rows);
    let mut stops: Vec<u32> = Vec::with_capacity(rows);
    let mut later: Vec<u16> = Vec::with_capacity(rows);
    let mut values: Vec<Values> = stores
        .iter()
        .map(|store| Values::with_capacity(*store, rows))
        .collect();
    fold_fragments(env, MEASURE_STREAM, &mut |_key, fragment| {
        let body = fragment
            .strip_prefix(MEASURE_MAGIC)
            .filter(|body| body.len() % row_bytes == 0)
            .ok_or_else(|| Error::invalid("vertex measure: a fragment of the wrong shape"))?;
        for row in body.chunks_exact(row_bytes) {
            let u32_at = |at: usize| u32::from_le_bytes(row[at..at + 4].try_into().expect("4"));
            let key =
                (u64::from(u32_at(0)) << 42) | (u64::from(u32_at(4)) << 21) | u64::from(u32_at(8));
            let index = u32::try_from(keys.len())
                .map_err(|_| Error::invalid("vertex measure: more than 2^32 vertices"))?;
            keys.push((key, index));
            stops.push(u32_at(12));
            later.push(u16::from_le_bytes(row[16..18].try_into().expect("2")));
            for (k, column) in values.iter_mut().enumerate() {
                let at = ROW_HEAD + 8 * k;
                column.push(f64::from_le_bytes(row[at..at + 8].try_into().expect("8")))?;
            }
        }
        Ok(())
    })?;
    if keys.len() != rows {
        return Err(Error::invalid(format!(
            "vertex measure: {} rows read and {rows} counted",
            keys.len()
        )));
    }
    // Positions are distinct (one row per skeleton voxel), so the key alone
    // orders the rows.
    keys.sort_unstable();
    let mask = (1u64 << 21) - 1;
    let coordinates = keys
        .iter()
        .map(|(key, _)| {
            [
                (key >> 42) as u32,
                ((key >> 21) & mask) as u32,
                (key & mask) as u32,
            ]
        })
        .collect();
    let order: Vec<u32> = keys.iter().map(|(_, index)| *index).collect();
    drop(keys);
    let stops = order.iter().map(|at| stops[*at as usize]).collect();
    let later = order.iter().map(|at| later[*at as usize]).collect();
    let values = values
        .into_iter()
        .map(|column| column.gather(1, &order))
        .collect();
    Ok(Measured {
        coordinates,
        stops,
        later,
        values,
    })
}

/// Hand the heap the plans freed back to the system, so that the resident
/// stages after them do not stack on the plans' high-water mark (glibc keeps
/// freed block buffers in its arenas).
fn release_freed_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free heap pages to the system.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Run one of the plans, profiled when asked.
fn run(
    assembly: &blockflow::assemble::Assembly,
    env: &ZarrEnvironment,
    plan: &PlanParams,
    keep: &[ImageId],
    name: &'static str,
) -> Result<()> {
    let listening = Listening::new(name, assembly);
    execute_phases(
        name,
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: plan.workers.max(1),
            keep_images: keep.iter().copied().collect(),
            ..Hints::default()
        },
        env,
        &listening.listeners(),
        &assembly.work(),
    )?;
    listening.finish();
    Ok(())
}

/// The vertex measure's block where none is given: measured fastest on the
/// osf-sa3x8 crop and region, with the lowest peak.
pub const MEASURE_BLOCK: [usize; 3] = [128, 256, 256];

/// The pack's block where none is given: the published labels' chunk, so
/// that each label chunk is read once.
pub const PACK_BLOCK: [usize; 3] = [64, 64, 128];

/// The plan `build` makes on `given`, else on `default` where one block of
/// it fits the memory budget, else on the grid the planner prices fastest
/// under the budget (`blockflow::strategy::choose_grid`).
///
/// The planner's choice is not the default because its model cannot price
/// these two phases yet: the vertex measure works through its block on every
/// worker (a plane each), which the model does not know, so it plans one
/// block per worker and holds that many halo windows (on the osf-sa3x8
/// region 2.9 GB at 24 s, where the default holds 1.8 GB at 21 s); and the
/// pack is bound by decoding the labels, which the model counts as one read
/// channel. Under a cap too small for one default block it is the planner's
/// grid that runs.
fn planned<T>(
    what: &str,
    volume: [usize; 3],
    given: Option<[usize; 3]>,
    default: [usize; 3],
    workers: usize,
    mut build: impl FnMut(BlockGrid) -> Result<(Assembly, T)>,
) -> Result<(Assembly, T)> {
    let host = Host::detect().with_workers(workers.max(1));
    let block = given.unwrap_or(default);
    let grid = BlockGrid::new(volume, block)?;
    let (assembly, built) = build(grid.clone())?;
    let needed = expected_bytes_per_block(&assembly.workflow.chain, &assembly.decomposition)
        .into_iter()
        .max()
        .unwrap_or(0);
    if given.is_some() || needed <= host.budget_bytes {
        eprintln!(
            "{what}: block {:?} ({}), {} blocks, {} MiB a block of {} MiB",
            grid.block(),
            if given.is_some() { "given" } else { "default" },
            grid.n_blocks(),
            needed >> 20,
            host.budget_bytes >> 20
        );
        return Ok((assembly, built));
    }
    drop((assembly, built));
    let started = Instant::now();
    let chosen = choose_grid(volume, &host, &CostModel::default(), build)?;
    let widest = chosen
        .active
        .iter()
        .map(|set| set.bytes())
        .max()
        .unwrap_or(0);
    eprintln!(
        "{what}: block {:?}, {} blocks, chosen of {} grids for {} workers ({:.2}s), the \
         default's {} MiB a block being over the budget; active set at most {} MiB of {} MiB",
        chosen.grid.block(),
        chosen.grid.n_blocks(),
        chosen.considered.len(),
        host.workers,
        started.elapsed().as_secs_f64(),
        needed >> 20,
        widest >> 20,
        host.budget_bytes >> 20
    );
    Ok((chosen.assembly, chosen.built))
}

/// Lease `bytes` of the process's memory budget for `what`, which stays
/// resident until the lease is dropped. Under a stated or cgroup cap a lease
/// that does not fit is refused, naming the cap that would hold it; under the
/// machine's own share it is let through with a warning.
pub fn lease_resident(bytes: u64, what: &str) -> Result<Option<Lease>> {
    let budget = process_budget();
    if let Some(lease) = budget.try_acquire_reclaiming(bytes) {
        return Ok(Some(lease));
    }
    let cap = process_cap();
    let held = budget.reserved_in_use();
    let needed = MemoryCap::needed_for(held.saturating_add(bytes));
    let message = format!(
        "{what} needs about {} MiB resident, and the memory cap ({cap}) leaves {} MiB beside \
         the {} MiB already held; a cap of about {} MiB holds it.",
        bytes.div_ceil(1 << 20),
        budget.free() >> 20,
        held >> 20,
        needed.div_ceil(1 << 20)
    );
    if cap.is_binding() {
        return Err(Error::invalid(message));
    }
    eprintln!("graph: warning: {message}");
    Ok(None)
}

/// Stages 1–3 over stored volumes: the raw graph and the columns
/// [`super::vessel_graph`] carries (radii, then `artery_raw`, `vein_raw`,
/// `artery_binary`, `vein_binary` where given), with the answers the resident
/// stages give; and the lease of the graph's resident bytes
/// ([`PlanParams::resident_bytes_per_vertex`]), taken once the vertices are
/// counted and before they are read.
pub fn measure_stored(
    inputs: &StoredInputs,
    params: &GraphParams,
    build: &BuildParams,
    plan: &PlanParams,
) -> Result<(RawGraph, Vec<Carried>, Option<Lease>)> {
    let (vessels_dtype, volume) = inputs.vessels.metadata()?;
    let check = |image: &AttachedImage, what: &str| -> Result<Dtype> {
        let (dtype, shape) = image.metadata()?;
        if shape != volume {
            return Err(Error::invalid(format!(
                "the {what} volume is {shape:?} and the vessel mask {volume:?}; every input is \
                 read at the skeleton's vertex coordinates."
            )));
        }
        Ok(dtype)
    };
    if volume.iter().any(|extent| *extent >= 1 << 21) {
        return Err(Error::invalid(format!(
            "a volume of {volume:?}; the vertex rows are keyed by 21 bits per axis"
        )));
    }
    // The pack's inputs: image 0 is the vessel mask, the other masks are
    // supplied; the measure's: image 0 is the packed masks, the channels are
    // supplied.
    let mut masks = vec![inputs.vessels.clone()];
    let mut channels = Vec::new();
    let supply = |list: &mut Vec<AttachedImage>, image: &AttachedImage, base: usize| {
        list.push(image.clone());
        ImageId::supplied(list.len() - 1 - base)
    };
    let mut packed_sources = vec![(
        supply(&mut masks, &inputs.skeleton, 1),
        check(&inputs.skeleton, "skeleton")?,
        SKELETON,
    )];
    let spacing = inputs.spacing;
    let mut signals = Vec::new();
    let mut names = Vec::new();
    // The signals, in the order the resident stages carry them.
    for (name, image, shift_vx, mask_bit) in [
        (
            "artery_raw",
            &inputs.artery_intensity,
            params.artery_shift_vx,
            None,
        ),
        ("vein_raw", &inputs.vein_intensity, 0.0, None),
        ("artery_binary", &inputs.artery_mask, 0.0, Some(ARTERY)),
        ("vein_binary", &inputs.vein_mask, 0.0, Some(VEIN)),
    ] {
        let Some(image) = image else {
            continue;
        };
        let dtype = check(image, name)?;
        let source = match mask_bit {
            Some(bit) => {
                packed_sources.push((supply(&mut masks, image, 1), dtype, bit));
                Source::Bit(bit)
            }
            None => Source::Channel(supply(&mut channels, image, 0), dtype),
        };
        signals.push(Signal {
            source,
            shift_um: shift_um(spacing, shift_vx)?,
        });
        let store = match mask_bit {
            Some(_) => Narrow::Flags,
            None => Narrow::of_dtype(dtype),
        };
        names.push((name, mask_bit.is_some(), store));
    }
    if plan.work.exists() {
        std::fs::remove_dir_all(&plan.work).map_err(Error::backend)?;
    }

    // The pack, with no chunk cache: every label chunk is read once.
    GRAPH.stage("graph plan: pack");
    let (assembly, packed) = planned(
        "graph pack",
        volume,
        plan.pack_block,
        PACK_BLOCK,
        plan.workers,
        |grid| {
            let mut builder = PlanBuilder::new(volume, vessels_dtype, grid);
            let packed = builder
                .pixels(Chain::op(PackMasksOp {
                    sources: packed_sources.clone(),
                }))?
                .image()?;
            Ok((builder.finish()?, packed))
        },
    )?;
    let pack_work = plan.work.join("pack");
    let env = ZarrEnvironment::attach(&pack_work, &masks)?.without_cache();
    run(&assembly, &env, plan, &[packed], "graph pack")?;
    drop((env, assembly));

    // The vertex measure over the packed masks and the channels.
    GRAPH.stage("graph plan: measure");
    let search = Arc::new(RadiusSearch::new(params.radius.clone())?);
    // The window reach: the walk's maximum, or the widest ball any signal
    // can be read over (the search radius of the longest stop), if larger;
    // at least one, for the neighbours.
    let longest: [f64; 3] =
        std::array::from_fn(|axis| params.radius.max_radius[axis] as f64 * spacing[axis]);
    let largest = signals
        .iter()
        .map(|signal| search_radius(longest, spacing, signal.shift_um))
        .fold(0.0f64, f64::max);
    let ball = Arc::new(Ball::new(largest)?);
    let reach = params
        .radius
        .max_radius
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(ball.radius())
        .max(1);
    let steps = forward_offsets(build.connectivity);
    if steps.len() > 16 {
        return Err(Error::invalid(format!(
            "{} later neighbours do not fit the 16 bits a vertex row holds.",
            steps.len()
        )));
    }
    let (assembly, ()) = planned(
        "graph measure",
        volume,
        plan.block,
        MEASURE_BLOCK,
        plan.workers,
        |grid| {
            let mut builder = PlanBuilder::new(volume, Dtype::U8, grid);
            builder.fragments(VertexMeasureOp {
                signals: signals.clone(),
                search: Arc::clone(&search),
                ball: Arc::clone(&ball),
                spacing,
                reach,
                steps: steps.clone(),
            })?;
            Ok((builder.finish()?, ()))
        },
    )?;
    let mut attached = vec![AttachedImage::at(
        pack_work.join(format!("level{}", packed.index())),
    )];
    attached.extend(channels);
    let env =
        ZarrEnvironment::attach(plan.work.join("measure"), &attached)?.with_cache(plan.cache_bytes);
    run(&assembly, &env, plan, &[], "graph measure")?;
    drop((assembly, ball));
    release_freed_memory();

    GRAPH.stage("graph plan: collect");
    let rows = count_measured(&env)?;
    let lease = match plan.resident_bytes_per_vertex {
        0 => None,
        per_vertex => lease_resident(
            (rows as u64).saturating_mul(per_vertex),
            &format!(
                "the vessel graph of {rows} raw vertices ({per_vertex} bytes a raw vertex at \
                 its widest)"
            ),
        )?,
    };
    let stores: Vec<Narrow> = names.iter().map(|(_, _, store)| *store).collect();
    let measured = collect_measured(&env, rows, &stores)?;
    drop(env);
    let _ = std::fs::remove_dir_all(&plan.work);
    release_freed_memory();

    GRAPH.stage("graph plan: raw graph");
    let Measured {
        coordinates,
        stops,
        later,
        values,
    } = measured;
    let raw = RawGraph::from_steps(volume, coordinates, &steps, &later)?;
    drop(later);
    let search = Arc::try_unwrap(search).unwrap_or_else(|shared| (*shared).clone());
    let mut carried = carried_stops(Arc::new(search), stops, spacing)?;
    for ((name, mask, _), values) in names.into_iter().zip(values) {
        let column = RawColumn::dense(1, values)?;
        carried.push(match mask {
            false => Carried::max(name, column),
            // A mask's maximum over a ball of its bit is 0 or 1 already.
            true => Carried::binary(name, column),
        });
    }
    Ok((raw, carried, lease))
}
