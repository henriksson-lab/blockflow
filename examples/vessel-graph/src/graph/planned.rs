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
//! itself (the skeleton's set voxels), merged in lexicographic order, and it
//! writes the skeleton's adjacent pairs of its core
//! (`blockflow::ops::adjacency::encode_adjacent_pairs`, merged by
//! `collect_pairs`), as [`super::build_blocked`] does.

use std::path::PathBuf;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::env::{BlockBuf, Environment};
use blockflow::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, SeamFold,
    SourceBlocks,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInput, SourceInputs};
use blockflow::ops::adjacency::{collect_pairs, encode_adjacent_pairs};
use blockflow::ops::Connectivity;
use blockflow::reach::Reach;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::AttachedImage;
use blockflow::{Dtype, Error, Result, ZarrEnvironment};
use rayon::prelude::*;

use crate::profile::{Listening, GRAPH};

use super::columns::Column;
use super::raw::{BuildParams, RawGraph};
use super::signal::{search_radius, shift_um, Ball, Reduce};
use super::vessel::Carried;
use super::{GraphParams, RadiusSearch};

const MEASURE_STREAM: &str = "graph.vertex-measure";
const PAIRS_STREAM: &str = "graph.edges";
const MEASURE_MAGIC: &[u8] = b"vessel-graph vertex measure v1";

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
    /// Block edge of the vertex measure, `z, y, x`.
    pub block: [usize; 3],
    /// Block edge of the pack, `z, y, x`: what holds the `uint64` labels.
    pub pack_block: [usize; 3],
    pub workers: usize,
    /// The chunk cache of the stored arrays, in bytes.
    pub cache_bytes: u64,
    /// Scratch directory for the packed masks; removed afterwards.
    pub work: PathBuf,
}

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
        1.0 + self.sources.len() as f64
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

/// Per skeleton voxel of the core: the radius walk's stop and each signal's
/// maximum over the vertex's ball; and the core's adjacent skeleton pairs.
/// See the module header.
pub struct VertexMeasureOp {
    signals: Vec<Signal>,
    search: RadiusSearch,
    ball: Ball,
    spacing: [f64; 3],
    reach: usize,
    connectivity: Connectivity,
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
            FragmentOutput::new(PAIRS_STREAM, Lifecycle::DeleteOnExit, Coverage::EveryBlock),
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
                        let stop = stopped.map_or(-1i64, |index| index as i64);
                        bytes.extend_from_slice(&stop.to_le_bytes());
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
        let pairs =
            encode_adjacent_pairs(skeleton.view(), origin, at.core, volume, self.connectivity)?;
        Ok(BlockOutput::fragment(MEASURE_STREAM, rows).with_fragment(PAIRS_STREAM, pairs))
    }
}

/// The rows of a finished run, in vertex (lexicographic) order.
struct Measured {
    coordinates: Vec<[usize; 3]>,
    stops: Vec<Option<usize>>,
    /// One column per signal.
    values: Vec<Vec<f64>>,
}

fn collect_measured(env: &dyn Environment, signals: usize) -> Result<Measured> {
    let row_bytes = 12 + 8 + 8 * signals;
    // Every row's bytes, as the blocks wrote them, and per row its position
    // packed into one sortable key.
    let mut bytes: Vec<u8> = Vec::new();
    let mut keys: Vec<(u64, u32)> = Vec::new();
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
        }
        bytes.extend_from_slice(body);
        Ok(())
    })?;
    // Positions are distinct (one row per skeleton voxel), so the key alone
    // orders the rows.
    keys.sort_unstable();
    let mut out = Measured {
        coordinates: Vec::with_capacity(keys.len()),
        stops: Vec::with_capacity(keys.len()),
        values: vec![Vec::with_capacity(keys.len()); signals],
    };
    for (_, index) in keys {
        let row = &bytes[index as usize * row_bytes..(index as usize + 1) * row_bytes];
        let u32_at = |at: usize| u32::from_le_bytes(row[at..at + 4].try_into().expect("4"));
        out.coordinates
            .push([u32_at(0) as usize, u32_at(4) as usize, u32_at(8) as usize]);
        let stop = i64::from_le_bytes(row[12..20].try_into().expect("8"));
        out.stops.push((stop >= 0).then_some(stop as usize));
        for (k, column) in out.values.iter_mut().enumerate() {
            column.push(f64::from_le_bytes(
                row[20 + 8 * k..28 + 8 * k].try_into().expect("8"),
            ));
        }
    }
    Ok(out)
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

/// Stages 1–3 over stored volumes: the raw graph and the columns
/// [`super::vessel_graph`] carries (radii, then `artery_raw`, `vein_raw`,
/// `artery_binary`, `vein_binary` where given), with the answers the resident
/// stages give.
pub fn measure_stored(
    inputs: &StoredInputs,
    params: &GraphParams,
    build: &BuildParams,
    plan: &PlanParams,
) -> Result<(RawGraph, Vec<Carried>)> {
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
        names.push((name, mask_bit.is_some()));
    }
    if plan.work.exists() {
        std::fs::remove_dir_all(&plan.work).map_err(Error::backend)?;
    }

    // The pack, with no chunk cache: every label chunk is read once.
    GRAPH.stage("graph plan: pack");
    let mut builder = PlanBuilder::new(
        volume,
        vessels_dtype,
        BlockGrid::new(volume, plan.pack_block)?,
    );
    let packed = builder
        .pixels(Chain::op(PackMasksOp {
            sources: packed_sources,
        }))?
        .image()?;
    let assembly = builder.finish()?;
    let pack_work = plan.work.join("pack");
    let env = ZarrEnvironment::attach(&pack_work, &masks)?.without_cache();
    run(&assembly, &env, plan, &[packed], "graph pack")?;
    drop(env);

    // The vertex measure over the packed masks and the channels.
    GRAPH.stage("graph plan: measure");
    let search = RadiusSearch::new(params.radius.clone())?;
    // The window reach: the walk's maximum, or the widest ball any signal
    // can be read over (the search radius of the longest stop), if larger;
    // at least one, for the pairs.
    let longest: [f64; 3] =
        std::array::from_fn(|axis| params.radius.max_radius[axis] as f64 * spacing[axis]);
    let largest = signals
        .iter()
        .map(|signal| search_radius(longest, spacing, signal.shift_um))
        .fold(0.0f64, f64::max);
    let ball = Ball::new(largest)?;
    let reach = params
        .radius
        .max_radius
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(ball.radius())
        .max(1);
    let mut builder = PlanBuilder::new(volume, Dtype::U8, BlockGrid::new(volume, plan.block)?);
    let measure = builder.fragments(VertexMeasureOp {
        signals: signals.clone(),
        search: search.clone(),
        ball,
        spacing,
        reach,
        connectivity: build.connectivity,
    })?;
    let assembly = builder.finish()?;
    let mut attached = vec![AttachedImage::at(
        pack_work.join(format!("level{}", packed.index())),
    )];
    attached.extend(channels);
    let env = ZarrEnvironment::attach(plan.work.join("measure"), &attached)?
        .with_cache(plan.cache_bytes);
    run(&assembly, &env, plan, &[], "graph measure")?;
    GRAPH.stage("graph plan: collect");
    let measured = collect_measured(&env, signals.len())?;
    let pairs = collect_pairs(&env, PAIRS_STREAM, measure.index(), volume)?;
    drop(env);
    let _ = std::fs::remove_dir_all(&plan.work);
    release_freed_memory();

    GRAPH.stage("graph plan: raw graph");
    let raw = RawGraph::assemble(volume, &measured.coordinates, &pairs)?;
    let mut carried = Vec::new();
    let radii = search.from_stops(&measured.stops, spacing)?;
    for (name, column) in radii.columns() {
        carried.push(Carried::max(name, column));
    }
    for ((name, mask), values) in names.into_iter().zip(measured.values) {
        carried.push(match mask {
            false => Carried::max(name, Column::scalar(values)),
            true => {
                let flags: Vec<bool> = values.iter().map(|value| *value != 0.0).collect();
                Carried::binary(name, Column::flags(&flags))
            }
        });
    }
    Ok((raw, carried))
}
