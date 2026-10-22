// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The `segment` and `tract` subcommands: OME-Zarr channels in, label images
//! published into the same image out.
//!
//! A run attaches level 0 of each channel, plans everything as one
//! `execute_phases` (two for a tract map: the clip range first), widens the
//! answers to `uint64` and publishes them with `ome_zarr::publish_label`, which
//! builds the label pyramid matching the image's levels. Nothing is loaded
//! whole.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::assemble::{Assembly, ImageId, PlanBuilder};
use blockflow::decomposition::CostModel;
use blockflow::geometry::BlockGrid;
use blockflow::ome_zarr::{publish_label, LabelPublication, OmeVolumePyramid};
use blockflow::op::Chain;
use blockflow::ops::voxelize::{ball, voxelize_into};
use blockflow::ops::WidenLabelsOp;
use blockflow::points::Point;
use blockflow::region::Region;
use blockflow::strategy::{choose_grid, execute_phases, Hints, Host};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::AttachedImage;
use blockflow::{Dtype, Error, Result, ZarrEnvironment};
use clap::Args;

use atlas_regions::Rounding;
use blockflow_atlas_registration::atlas_space::{LookupParams, OpenParams, RegisteredAtlas};

use crate::graph::cli::{atlas_lattice, image_frame, level_lattice, write_density};
use crate::profile::{Listening, Profile};

use super::clip::ClipParams;
use super::crop::write_crop;
use super::deep_fill::DeepFillSetup;
use super::params::{ChannelOverrides, ChannelRole, PostOverrides, SegmentOverrides};
use super::pipeline::{append_segment_phases, append_segment_phases_unthinned, SegmentParams};
use super::plan::{workers, Cursor, Image};
use super::skeleton::SkeletonParams;
use super::thin::{round_plan, thin_stored, ThinPlan};
use super::tract::{
    append_clip_range_phases, append_tract_phases, count_points, for_each_point_slab,
    read_clip_range, tract_binarize, Rescale, TractParams,
};

fn triple<T: Copy>(values: &[T], what: &str) -> Result<[T; 3]> {
    values
        .try_into()
        .map_err(|_| Error::invalid(format!("{what} needs three values, z,y,x")))
}

fn pair(values: &[f64], what: &str) -> Result<(f64, f64)> {
    match values {
        [low, high] => Ok((*low, *high)),
        _ => Err(Error::invalid(format!("{what} needs two values, low,high"))),
    }
}

/// The input every subcommand shares.
#[derive(Debug, Clone, Args)]
pub struct InputArgs {
    /// OME-Zarr image (a local copy: label images are written into it).
    #[arg(long)]
    pub zarr: PathBuf,
    /// Time point.
    #[arg(long, default_value_t = 0)]
    pub time: usize,
    /// Block edge, z,y,x. Without it the planner chooses the block grid for
    /// the volume, the plan and the worker count
    /// (`blockflow::strategy::choose_grid`).
    #[arg(long, value_delimiter = ',')]
    pub block: Option<Vec<usize>>,
    /// Worker threads (default: one per core).
    #[arg(long)]
    pub workers: Option<usize>,
    /// Scratch directory for the run's intermediate images (default: next to
    /// the image). Removed afterwards.
    #[arg(long)]
    pub work: Option<PathBuf>,
    /// Run on a crop of level 0, `z0:z1,y0:y1,x0:x1` (Python slices, `:` for a
    /// whole axis): the crop of the channels is copied to `--crop-out` (in the
    /// source's physical frame, channels renumbered from 0 in `--channel`
    /// order) and the labels are published there.
    #[arg(long, requires = "crop_out", allow_hyphen_values = true)]
    pub crop: Option<String>,
    /// The cropped image `--crop` writes, replacing what is there.
    #[arg(long, requires = "crop")]
    pub crop_out: Option<PathBuf>,
}

impl InputArgs {
    fn work_dir(&self, what: &str) -> PathBuf {
        self.work.clone().unwrap_or_else(|| {
            let name = self
                .zarr
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "image".to_string());
            self.zarr
                .with_file_name(format!(".{name}.vessel-graph-{what}-work"))
        })
    }

    fn workers(&self) -> usize {
        self.workers.unwrap_or_else(workers).max(1)
    }

    fn block(&self) -> Result<Option<[usize; 3]>> {
        self.block
            .as_deref()
            .map(|block| triple(block, "--block"))
            .transpose()
    }

    /// Without `--block`, a cursor that writes its labels on [`LABEL_BLOCK`].
    fn labelled<'a>(&self, c: Cursor<'a>, volume: [usize; 3]) -> Result<Cursor<'a>> {
        Ok(match self.block {
            Some(_) => c,
            None => c.labelling_on(BlockGrid::new(volume, LABEL_BLOCK)?),
        })
    }

    /// The plan `build` makes on `--block`, or on the grid the planner prices
    /// fastest for this volume on `--workers` (and the machine's memory).
    fn planned<T>(
        &self,
        what: &str,
        volume: [usize; 3],
        mut build: impl FnMut(BlockGrid) -> Result<(Assembly, T)>,
    ) -> Result<(BlockGrid, Assembly, T)> {
        if let Some(block) = self.block()? {
            let grid = BlockGrid::new(volume, block)?;
            let (assembly, built) = build(grid.clone())?;
            eprintln!(
                "{what}: block {:?} (--block), {} blocks",
                grid.block(),
                grid.n_blocks()
            );
            return Ok((grid, assembly, built));
        }
        let started = Instant::now();
        let host = Host::detect().with_workers(self.workers());
        let chosen = choose_grid(volume, &host, &CostModel::default(), build)?;
        let widest = chosen
            .active
            .iter()
            .map(|set| set.bytes())
            .max()
            .unwrap_or(0);
        eprintln!(
            "{what}: block {:?}, {} blocks, chosen of {} grids for {} workers ({:.2}s); \
             active set at most {} MiB of {} MiB",
            chosen.grid.block(),
            chosen.grid.n_blocks(),
            chosen.considered.len(),
            host.workers,
            started.elapsed().as_secs_f64(),
            widest >> 20,
            host.budget_bytes >> 20
        );
        if crate::profile::enabled() {
            for (phase, set) in chosen.active.iter().enumerate() {
                eprintln!(
                    "active {what}: {phase:>2} {:>7.1} MiB x {:>2} in flight = {:>7.1} MiB",
                    set.per_block as f64 / (1u64 << 20) as f64,
                    set.in_flight,
                    set.bytes() as f64 / (1u64 << 20) as f64
                );
            }
        }
        Ok((chosen.grid, chosen.assembly, chosen.built))
    }

    /// With `--crop`: copy the crop of `channels` and return the arguments
    /// that run on the copy, whose channels are `0..channels.len()`.
    fn cropped(&self, channels: &[usize]) -> Result<Option<InputArgs>> {
        let (Some(crop), Some(out)) = (&self.crop, &self.crop_out) else {
            return Ok(None);
        };
        let started = Instant::now();
        let written = write_crop(&self.zarr, channels, self.time, crop, out)?;
        eprintln!(
            "crop: {:?} at {:?} of {} into {} ({:.1}s)",
            written.shape,
            written.start,
            self.zarr.display(),
            out.display(),
            started.elapsed().as_secs_f64()
        );
        Ok(Some(InputArgs {
            zarr: out.clone(),
            time: 0,
            crop: None,
            crop_out: None,
            ..self.clone()
        }))
    }
}

/// `segment`: vessel channel(s) to a binary mask and its skeleton.
///
/// Each channel is binarised with its role's defaults (`--role`), then the
/// parameter file (`--params`), then the flags below, which apply to every
/// channel. See [`super::params`] for the roles and the file.
#[derive(Debug, Clone, Args)]
pub struct SegmentArgs {
    #[command(flatten)]
    pub input: InputArgs,
    /// Channels to binarise and unite (repeat, or comma-separate).
    #[arg(long, value_delimiter = ',', default_values_t = vec![0])]
    pub channel: Vec<usize>,
    /// Each channel's role, in `--channel` order: vessel, artery or vein
    /// (default: the first channel vessel, every other artery). The role picks
    /// the channel's clip range, thresholds and arms.
    #[arg(long, value_delimiter = ',')]
    pub role: Vec<String>,
    /// JSON file of per-channel overrides on top of the roles' defaults.
    #[arg(long)]
    pub params: Option<PathBuf>,
    /// Label image names are `<name>-mask` and `<name>-skeleton`, and with
    /// several channels `<name>-<role>-mask` per channel.
    #[arg(long, default_value = "vessel")]
    pub name: String,
    /// Raw intensity range clipped onto the bins, low,high, for every channel
    /// (default: the role's).
    #[arg(long, value_delimiter = ',')]
    pub clip_range: Option<Vec<f64>>,
    /// Bins the clip range is spread over (default 4096).
    #[arg(long)]
    pub bins: Option<u32>,
    #[arg(long)]
    pub no_lightsheet: bool,
    #[arg(long)]
    pub no_median: bool,
    #[arg(long)]
    pub no_deconvolve: bool,
    #[arg(long)]
    pub no_adaptive: bool,
    #[arg(long)]
    pub no_equalize: bool,
    #[arg(long)]
    pub no_vesselize: bool,
    /// Deconvolution residual threshold, in bins (vessel 750, artery/vein 450).
    #[arg(long)]
    pub deconvolve_threshold: Option<f64>,
    /// Equalised intensity threshold (vessel 1.1).
    #[arg(long)]
    pub equalize_threshold: Option<f64>,
    /// Tubeness threshold (vessel 120).
    #[arg(long)]
    pub vesselize_threshold: Option<f64>,
    /// Smoothing passes after each binarisation and after the union; 0 for none
    /// (default 6).
    #[arg(long)]
    pub smooth_passes: Option<usize>,
    /// Do not fill holes.
    #[arg(long)]
    pub no_fill: bool,
    /// Keep the volume's faces when skeletonising.
    #[arg(long)]
    pub keep_border: bool,
    /// The thinning runs resident (one iterative phase holding two bytes a
    /// voxel) when that is at most this many MiB, and over stored images, a
    /// round per phase, otherwise (default: 8192, or a quarter of the memory
    /// cap's usable bytes when that is less). The skeleton is the same either
    /// way; the stored rounds take more wall time.
    #[arg(long)]
    pub thin_memory_mb: Option<u64>,
    /// Skip the deep vessel filling (the CNN that fills hollow lumens after
    /// each channel's smooth and fill). It is on by default in a build with
    /// the `deep-fill` feature, as in the reference's configuration, and off
    /// without it.
    #[arg(long, conflicts_with = "deep_fill_weights")]
    pub no_deep_fill: bool,
    /// The deep-filling network's weights, a safetensors file converted from
    /// the reference's `cnn_binary_vessel_filling.pth` by
    /// `reference-clearmap/convert_weights.py` (default: $VESSEL_FILL_WEIGHTS).
    /// Giving it switches the deep filling on.
    #[arg(long)]
    pub deep_fill_weights: Option<PathBuf>,
    /// Deep-filling probability threshold (default 0.5).
    #[arg(long)]
    pub deep_fill_threshold: Option<f64>,
    /// Block edge of the deep-filling phases, z,y,x; each block runs through
    /// the network whole, with a halo of 64 voxels (128 at resample 2).
    #[arg(long, value_delimiter = ',', default_values_t = vec![256, 512, 512])]
    pub deep_fill_block: Vec<usize>,
}

impl SegmentArgs {
    /// The channels' roles: `--role`, else the parameter file's, else the
    /// default for the position.
    pub fn roles(&self, file: &SegmentOverrides) -> Result<Vec<ChannelRole>> {
        let mut roles = file.roles(self.channel.len());
        if !self.role.is_empty() {
            if self.role.len() != self.channel.len() {
                return Err(Error::invalid(format!(
                    "{} roles for {} channels; give one role per channel",
                    self.role.len(),
                    self.channel.len()
                )));
            }
            roles = self
                .role
                .iter()
                .map(|role| ChannelRole::parse(role))
                .collect::<Result<_>>()?;
        }
        Ok(roles)
    }

    fn overrides(&self) -> Result<SegmentOverrides> {
        match &self.params {
            Some(path) => SegmentOverrides::read(path),
            None => Ok(SegmentOverrides::default()),
        }
    }

    /// The flags, as overrides of every channel.
    fn flag_overrides(&self) -> Result<ChannelOverrides> {
        let off = |set: bool| set.then_some(false);
        Ok(ChannelOverrides {
            role: None,
            clip_range: self
                .clip_range
                .as_deref()
                .map(|range| pair(range, "--clip-range"))
                .transpose()?,
            bins: self.bins,
            lightsheet: off(self.no_lightsheet),
            median: off(self.no_median),
            deconvolve: off(self.no_deconvolve),
            adaptive: off(self.no_adaptive),
            equalize: off(self.no_equalize),
            vesselize: off(self.no_vesselize),
            deconvolve_threshold: self.deconvolve_threshold,
            equalize_threshold: self.equalize_threshold,
            vesselize_threshold: self.vesselize_threshold,
            smooth_passes: self.smooth_passes,
            fill: off(self.no_fill),
            deep_fill: match (self.no_deep_fill, &self.deep_fill_weights) {
                (true, _) => Some(false),
                (false, Some(_)) => Some(true),
                (false, None) => None,
            },
            deep_fill_resample: None,
            deep_fill_threshold: self.deep_fill_threshold,
        })
    }

    /// The parameters and roles these arguments describe.
    pub fn params_and_roles(&self) -> Result<(SegmentParams, Vec<ChannelRole>)> {
        let file = self.overrides()?;
        let roles = self.roles(&file)?;
        let mut params = file.params(&roles)?;
        let flags = self.flag_overrides()?;
        for (channel, role) in params.channels.iter_mut().zip(&roles) {
            // A threshold flag names an arm the large-vessel roles leave off;
            // it applies where the arm is on.
            let mut flags = flags.clone();
            let b = &channel.binarize;
            if b.equalize.is_none() {
                flags.equalize_threshold = None;
            }
            if b.vesselize.is_none() {
                flags.vesselize_threshold = None;
            }
            if b.deconvolve.is_none() {
                flags.deconvolve_threshold = None;
            }
            if channel.deep_fill.is_none() && flags.deep_fill != Some(true) {
                flags.deep_fill_threshold = None;
            }
            flags.apply(channel, *role).map_err(|error| {
                Error::invalid(format!("channel role {}: {error}", role.name()))
            })?;
            channel.binarize.validate()?;
        }
        PostOverrides {
            smooth_passes: self.smooth_passes,
            fill: (self.no_fill).then_some(false),
        }
        .apply(&mut params.combined);
        params.skeleton = SkeletonParams {
            delete_border: !self.keep_border,
            ..SkeletonParams::default()
        };
        params.deep_fill = DeepFillSetup {
            weights: self.deep_fill_weights.clone(),
            block: triple(&self.deep_fill_block, "--deep-fill-block")?,
        };
        Ok((params, roles))
    }

    /// The parameters these arguments describe.
    pub fn params(&self) -> Result<SegmentParams> {
        Ok(self.params_and_roles()?.0)
    }
}

/// The label name of each channel's own mask: `<name>-<role>-mask`, with the
/// channel index after the role where two channels share one.
pub fn channel_mask_labels(name: &str, roles: &[ChannelRole], channels: &[usize]) -> Vec<String> {
    roles
        .iter()
        .zip(channels)
        .map(|(role, channel)| {
            let shared = roles.iter().filter(|other| *other == role).count() > 1;
            match shared {
                true => format!("{name}-{}{channel}-mask", role.name()),
                false => format!("{name}-{}-mask", role.name()),
            }
        })
        .collect()
}

/// What a `segment` run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentSummary {
    pub mask_label: String,
    pub skeleton_label: String,
    /// Each channel's own mask, with two channels or more.
    pub channel_mask_labels: Vec<String>,
    pub phases: usize,
}

fn open_channels(input: &InputArgs, channels: &[usize]) -> Result<Vec<OmeVolumePyramid>> {
    if channels.is_empty() {
        return Err(Error::invalid("no --channel given"));
    }
    channels
        .iter()
        .map(|&channel| OmeVolumePyramid::open(&input.zarr, channel, input.time))
        .collect()
}

fn fresh(work: &Path) -> Result<()> {
    if work.exists() {
        fs::remove_dir_all(work).map_err(Error::backend)?;
    }
    Ok(())
}

fn hints(input: &InputArgs, keep: &[ImageId]) -> Hints {
    Hints {
        concurrency: input.workers(),
        keep_images: keep.iter().copied().collect::<BTreeSet<_>>(),
        ..Hints::default()
    }
}

fn publish(
    pyramid: &OmeVolumePyramid,
    work: &Path,
    image: Image,
    name: &str,
    workers: usize,
) -> Result<()> {
    publish_label(
        pyramid,
        LabelPublication {
            name,
            level0: &work.join(format!("level{}", image.id.index())),
            image_label: None,
            table: None,
            replace: true,
            pyramid_blocks: Vec::new(),
        },
        workers,
    )
}

/// Run `segment`: binarise, unite, post-process and skeletonise, and publish
/// `<name>-mask` and `<name>-skeleton` into the image.
pub fn run_segment(args: &SegmentArgs) -> Result<SegmentSummary> {
    if let Some(input) = args.input.cropped(&args.channel)? {
        return run_segment(&SegmentArgs {
            input,
            channel: (0..args.channel.len()).collect(),
            ..args.clone()
        });
    }
    let started = Instant::now();
    let (params, roles) = args.params_and_roles()?;
    let pyramids = open_channels(&args.input, &args.channel)?;
    let attached = pyramids
        .iter()
        .map(|pyramid| pyramid.attached_level(0))
        .collect::<Result<Vec<_>>>()?;
    let (dtype, volume) = attached[0].metadata()?;
    let mut sources = vec![Image::new(0usize, dtype)];
    for (index, image) in attached.iter().enumerate().skip(1) {
        let (other_dtype, other_volume) = image.metadata()?;
        if other_volume != volume {
            return Err(Error::invalid(format!(
                "channel {} is {other_volume:?} and channel {} is {volume:?}; the channels of \
                 one segmentation share a lattice",
                args.channel[index], args.channel[0]
            )));
        }
        sources.push(Image::new(ImageId::supplied(index - 1), other_dtype));
    }

    // The thinning in the plan, as one iterative phase holding the volume
    // twice, when that fits `--thin-memory-mb`; otherwise everything but the
    // thinning, which then runs over the stored image the plan leaves
    // (`thin::thin_stored`).
    let voxels = volume.iter().product::<usize>() as u64;
    let resident_thinning = 2 * voxels <= thin_memory_bytes(args.thin_memory_mb);
    let (grid, assembly, planned) = args.input.planned("segment", volume, |grid| {
        let mut plan = PlanBuilder::new(volume, dtype, grid);
        let mut c = Cursor::new(&mut plan, sources[0]);
        if args.input.block.is_none() {
            c = c.pricing_iterations();
        }
        let mut c = args.input.labelled(c, volume)?;
        let segmented = match resident_thinning {
            true => append_segment_phases(&mut c, &sources, &params)?,
            false => append_segment_phases_unthinned(&mut c, &sources, &params)?,
        };
        // The labels are written on the publication grid, which sets their
        // chunk shape, whatever grid computed them.
        c.to_label_grid();
        let mask = c.pixels(c.on(segmented.mask, vec![Chain::op(WidenLabelsOp)]))?;
        // The skeleton as a label (resident thinning), or the image the stored
        // thinning starts from.
        let skeleton = match resident_thinning {
            true => c.pixels(c.on(segmented.skeleton, vec![Chain::op(WidenLabelsOp)]))?,
            false => segmented.skeleton,
        };
        // With several channels, each channel's own mask too: the large-vessel
        // masks are what the graph's artery/vein columns read.
        let channel_masks = match segmented.channels.len() {
            1 => Vec::new(),
            _ => segmented
                .channels
                .iter()
                .map(|channel| c.pixels(c.on(channel.processed, vec![Chain::op(WidenLabelsOp)])))
                .collect::<Result<Vec<_>>>()?,
        };
        Ok((plan.finish()?, (mask, skeleton, channel_masks)))
    })?;
    let (mask, skeleton, channel_masks) = planned;
    let channel_labels = match channel_masks.is_empty() {
        true => Vec::new(),
        false => channel_mask_labels(&args.name, &roles, &args.channel),
    };

    let work = args.input.work_dir("segment");
    fresh(&work)?;
    let env = ZarrEnvironment::attach(&work, &attached)?;
    eprintln!(
        "segment: {volume:?} {}, {} channel(s) ({}), {} phases",
        dtype.numpy_name(),
        sources.len(),
        roles
            .iter()
            .map(|role| role.name())
            .collect::<Vec<_>>()
            .join(", "),
        assembly.n_phases()
    );
    let deep_filled: Vec<String> = roles
        .iter()
        .zip(&params.channels)
        .filter_map(|(role, channel)| {
            channel.deep_fill.map(|d| {
                format!(
                    "{} (resample {}, threshold {})",
                    role.name(),
                    d.resample,
                    d.threshold
                )
            })
        })
        .collect();
    if !deep_filled.is_empty() {
        eprintln!("segment: deep fill of {}", deep_filled.join(", "));
    }
    let listening = Listening::new("segment", &assembly);
    let stats = execute_phases(
        "vessel-graph segment",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(
            &args.input,
            &[mask.id, skeleton.id]
                .into_iter()
                .chain(channel_masks.iter().map(|image| image.id))
                .collect::<Vec<_>>(),
        ),
        &env,
        &listening.listeners(),
        &assembly.work(),
    )?;
    listening.finish();
    drop((stats, env));

    // The stored thinning, then its answer widened to a label in a plan of
    // its own.
    let widen_work = work.with_extension("skeleton");
    let skeleton = match resident_thinning {
        true => (work.clone(), skeleton),
        false => {
            let thin_work = work.with_extension("thin");
            // The rounds' own grid: the thinning reaches as far on every axis,
            // which is rarely what suits the windows before it.
            let passes = params.skeleton.passes_per_round;
            let (thin_grid, _, _) = args.input.planned("thinning", volume, |grid| {
                Ok((round_plan(volume, grid, passes)?, ()))
            })?;
            let thinned = thin_stored(
                &AttachedImage::at(work.join(format!("level{}", skeleton.id.index()))),
                &ThinPlan {
                    grid: thin_grid.block(),
                    rounds_per_plan: THIN_ROUNDS_PER_PLAN,
                    workers: args.input.workers(),
                    round_limit: params.skeleton.round_limit,
                    passes: params.skeleton.passes_per_round,
                    work: thin_work.clone(),
                    cache_bytes: Some(THIN_CACHE_BYTES),
                },
            )?;
            eprintln!(
                "segment: thinned over stored images in {} rounds",
                thinned.rounds
            );
            let label_grid = match args.input.block {
                Some(_) => grid.clone(),
                None => BlockGrid::new(volume, LABEL_BLOCK)?,
            };
            let mut plan = PlanBuilder::new(volume, Dtype::Bool, label_grid);
            let widened = plan.pixels(Chain::op(WidenLabelsOp))?.image()?;
            let assembly = plan.finish()?;
            fresh(&widen_work)?;
            let env = ZarrEnvironment::attach(&widen_work, &[AttachedImage::at(&thinned.level0)])?;
            execute_phases(
                "vessel-graph skeleton",
                &assembly.workflow,
                &assembly.decomposition,
                &hints(&args.input, &[widened]),
                &env,
                &[],
                &assembly.work(),
            )?;
            drop(env);
            let _ = fs::remove_dir_all(&thin_work);
            (widen_work.clone(), Image::new(widened, Dtype::U64))
        }
    };

    let mask_label = format!("{}-mask", args.name);
    let skeleton_label = format!("{}-skeleton", args.name);
    let workers = args.input.workers();
    let post = Profile::new("segment");
    post.stage("publish labels");
    publish(&pyramids[0], &work, mask, &mask_label, workers)?;
    publish(
        &pyramids[0],
        &skeleton.0,
        skeleton.1,
        &skeleton_label,
        workers,
    )?;
    for (image, label) in channel_masks.iter().zip(&channel_labels) {
        publish(&pyramids[0], &work, *image, label, workers)?;
    }
    let _ = fs::remove_dir_all(&work);
    let _ = fs::remove_dir_all(&widen_work);
    post.finish();
    eprintln!("segment: done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(SegmentSummary {
        mask_label,
        skeleton_label,
        channel_mask_labels: channel_labels,
        phases: assembly.n_phases(),
    })
}

/// The block the published labels are written on when the planner chooses the
/// compute grid. A label's chunk shape follows the block that wrote it
/// (`zarr_env::chunk_for_block`: `[64, 64, 128]` for `uint64` here), and that
/// is a layout for its readers — `graph`'s blocks, a viewer — rather than a
/// compute choice, so it does not move with the grid the segmentation ran on.
/// It is the grid `--block` defaulted to before the planner chose.
const LABEL_BLOCK: [usize; 3] = [64, 256, 256];

/// The resident thinning's ceiling: `--thin-memory-mb`, else
/// [`THIN_MEMORY_BYTES`] or a quarter of the memory cap's usable bytes,
/// whichever is less. The resident phase holds its two buffers for the whole
/// iteration, beside the blocks in flight, so it may take only part of the cap.
fn thin_memory_bytes(stated_mb: Option<u64>) -> u64 {
    match stated_mb {
        Some(mb) => mb.saturating_mul(1 << 20),
        None => THIN_MEMORY_BYTES.min(blockflow::budget::process_cap().usable_bytes() / 4),
    }
}

/// The resident thinning's default ceiling: a volume of 4 G voxels, a whole
/// tile but not a whole brain.
const THIN_MEMORY_BYTES: u64 = 8192 << 20;

/// Thinning rounds per plan: a convergence test after each plan, and up to
/// this many rounds run past the fixed point (copies, every window unchanged).
const THIN_ROUNDS_PER_PLAN: usize = 8;

/// The stored thinning's chunk cache: enough for the windows of the blocks in
/// flight. The environment's default (an eighth of the machine) was slower on
/// the osf-sa3x8 crop.
const THIN_CACHE_BYTES: u64 = 1 << 30;

/// The points mapped into the atlas at a time, so that the mapping's own
/// arrays stay bounded whatever the mask holds.
const POINT_CHUNK: usize = 1 << 20;

/// The density of `counts` points per atlas voxel (row-major on `shape`):
/// a ball of radius `radius` per point, as `RegisteredAtlas::density` renders
/// them, from one weighted point per voxel holding any.
fn count_density(counts: &[u32], shape: [usize; 3], radius: [usize; 3]) -> Result<Voxels> {
    let mut placed = Vec::new();
    for (index, count) in counts.iter().enumerate() {
        if *count > 0 {
            let at = [
                index / (shape[1] * shape[2]),
                index / shape[2] % shape[1],
                index % shape[2],
            ];
            placed.push(Point::weighted(at, f64::from(*count)));
        }
    }
    let grid = BlockGrid::whole(shape)?;
    let mut out = ndarray::Array3::<f64>::zeros((shape[0], shape[1], shape[2]));
    voxelize_into(
        &[([0, 0, 0], placed)],
        &grid,
        &ball(radius),
        &Region::whole(&shape),
        out.view_mut(),
    )?;
    Ok(Voxels::from(out))
}

/// `tract`: a tract channel to a mask and a point density.
#[derive(Debug, Clone, Args)]
pub struct TractArgs {
    #[command(flatten)]
    pub input: InputArgs,
    #[arg(long, default_value_t = 0)]
    pub channel: usize,
    /// Label image names are `<name>-mask` and `<name>-density`.
    #[arg(long, default_value = "tract")]
    pub name: String,
    /// The clip range's percentiles of the raw intensities, low,high.
    #[arg(long, value_delimiter = ',', default_values_t = vec![70.0, 99.999])]
    pub percentiles: Vec<f64>,
    /// Sample one voxel in this many for the percentiles.
    #[arg(long, default_value_t = 1000)]
    pub sample_every: usize,
    /// Density ball radius, z,y,x voxels.
    #[arg(long, value_delimiter = ',', default_values_t = vec![5, 5, 5])]
    pub density_radius: Vec<usize>,
    /// Also write the mask's points to this CSV (z,y,x per row).
    #[arg(long)]
    pub points_csv: Option<PathBuf>,
    /// Scale the CSV's points from the image's shape to this one, z,y,x.
    #[arg(long, value_delimiter = ',', conflicts_with = "registration")]
    pub scale_to: Option<Vec<usize>>,
    /// Map the mask's points into atlas space through this registration
    /// directory (see `graph --registration`): the CSV gains `atlas_z,
    /// atlas_y, atlas_x, annotation, hemisphere`, and `--atlas-density`
    /// voxelises them on the atlas grid.
    #[arg(long)]
    pub registration: Option<PathBuf>,
    /// The prepared atlas the registration used as its moving image (default:
    /// that moving image, if it is a prepared atlas).
    #[arg(long, requires = "registration")]
    pub atlas: Option<PathBuf>,
    /// Do not compare the atlas's reference with the registration's moving
    /// image voxel for voxel.
    #[arg(long, requires = "registration")]
    pub skip_atlas_check: bool,
    /// Collapse the points' regions onto their ancestors at this tree depth.
    #[arg(long, requires = "registration")]
    pub level: Option<i64>,
    /// Write the atlas-space density (points per `--density-radius` ball of
    /// atlas voxels) as a one-level `float32` OME-Zarr image here.
    #[arg(long, requires = "registration")]
    pub atlas_density: Option<PathBuf>,
}

/// What a `tract` run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct TractSummary {
    pub range: (f64, f64),
    pub points: usize,
    pub mask_label: String,
    pub density_label: String,
    /// With `--registration`: points mapped outside the atlas (region 0).
    pub atlas_outside: Option<usize>,
}

/// Run `tract`: measure the clip range, binarise, voxelise, and publish
/// `<name>-mask` and `<name>-density` into the image.
pub fn run_tract(args: &TractArgs) -> Result<TractSummary> {
    if let Some(input) = args.input.cropped(&[args.channel])? {
        return run_tract(&TractArgs {
            input,
            channel: 0,
            ..args.clone()
        });
    }
    let params = TractParams {
        percentiles: pair(&args.percentiles, "--percentiles")?,
        sample_every: args.sample_every,
        binarize: tract_binarize(ClipParams::default()),
        density_radius: triple(&args.density_radius, "--density-radius")?,
    };
    params.validate()?;
    let pyramid = OmeVolumePyramid::open(&args.input.zarr, args.channel, args.input.time)?;
    let registered = args
        .registration
        .as_deref()
        .map(|dir| {
            let params = OpenParams {
                atlas: args.atlas.clone(),
                check_reference: !args.skip_atlas_check,
            };
            RegisteredAtlas::open(dir, &params)
        })
        .transpose()?;
    if registered.is_some() && args.points_csv.is_none() && args.atlas_density.is_none() {
        return Err(Error::invalid(
            "--registration maps the points into the atlas for --points-csv and \
             --atlas-density; give at least one",
        ));
    }
    let level0 = level_lattice(&pyramid.levels()[0]);
    let attached = pyramid.attached_level(0)?;
    let (dtype, volume) = attached.metadata()?;
    let source = Image::new(0usize, dtype);
    let work = args.input.work_dir("tract");
    // One grid for both runs, chosen (without `--block`) on the second, which
    // is where the work is.
    let (grid, assembly, images) = args.input.planned("tract", volume, |grid| {
        let mut plan = PlanBuilder::new(volume, dtype, grid);
        let mut c = args
            .input
            .labelled(Cursor::new(&mut plan, source), volume)?;
        // The range is only known after run 1; any range plans the same phases.
        let images = append_tract_phases(&mut c, source, &params, (0.0, 1.0))?;
        Ok((plan.finish()?, images))
    })?;
    drop((assembly, images));

    // Run 1: the clip range.
    let mut plan = PlanBuilder::new(volume, dtype, grid.clone());
    append_clip_range_phases(&mut plan, source, &params)?;
    let assembly = plan.finish()?;
    fresh(&work)?;
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&attached))?;
    let listening = Listening::new("tract range", &assembly);
    execute_phases(
        "vessel-graph tract range",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(&args.input, &[]),
        &env,
        &listening.listeners(),
        &assembly.work(),
    )?;
    listening.finish();
    let range = read_clip_range(&env)?;
    drop(env);
    eprintln!("tract: clip range {range:?}");

    // Run 2: mask, points, density.
    let mut plan = PlanBuilder::new(volume, dtype, grid);
    let mut c = args
        .input
        .labelled(Cursor::new(&mut plan, source), volume)?;
    let images = append_tract_phases(&mut c, source, &params, range)?;
    let assembly = plan.finish()?;
    fresh(&work)?;
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&attached))?;
    let listening = Listening::new("tract", &assembly);
    execute_phases(
        "vessel-graph tract",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(&args.input, &[images.mask.id, images.density.id]),
        &env,
        &listening.listeners(),
        &assembly.work(),
    )?;
    listening.finish();
    let post = Profile::new("tract");
    post.stage("points: map, label, csv");
    let point_count = count_points(&env)?;
    // The points, streamed one row of blocks at a time (each row sorted, the
    // rows in z order: the whole list's order), in chunks of at most
    // `POINT_CHUNK` for the mapping.
    let mut atlas_outside = None;
    let mut csv = args
        .points_csv
        .as_ref()
        .map(|path| {
            fs::File::create(path)
                .map(std::io::BufWriter::new)
                .map_err(Error::backend)
        })
        .transpose()?;
    if let Some(registered) = &registered {
        let image = image_frame(pyramid.image_root(), level0);
        let to_sample = registered.image_to_sample(&image, [0.0; 3])?;
        if let Some(warning) = registered.frame_warning(&to_sample, volume) {
            eprintln!("tract: warning: {warning}");
        }
        let lookup = LookupParams {
            level: args.level,
            rounding: Rounding::Nearest,
        };
        // Whether the rows carry a hemisphere: the atlas's, whatever the points.
        let hemispheres = registered.lookup(&[], &lookup)?.hemisphere.is_some();
        if let Some(out) = &mut csv {
            write!(out, "z,y,x,atlas_z,atlas_y,atlas_x,annotation").map_err(Error::backend)?;
            writeln!(out, "{}", if hemispheres { ",hemisphere" } else { "" })
                .map_err(Error::backend)?;
        }
        // The atlas density as a count per atlas voxel: a ball of weight `c`
        // where `c` points fall is the sum of their `c` unit balls, exactly
        // (integer counts in `f64`).
        let shape = registered.shape();
        let mut counts = args
            .atlas_density
            .is_some()
            .then(|| vec![0u32; shape[0] * shape[1] * shape[2]]);
        let mut mapped_count = 0usize;
        let mut outside = 0usize;
        for_each_point_slab(&env, &mut |slab| {
            for points in slab.chunks(POINT_CHUNK) {
                let image_points: Vec<[f64; 3]> =
                    points.iter().map(|p| p.map(|v| v as f64)).collect();
                let atlas = registered.map(&to_sample, &image_points).atlas;
                drop(image_points);
                let columns = registered.lookup(&atlas, &lookup)?;
                outside += registered.outside(&atlas, Rounding::Nearest);
                mapped_count += atlas.len();
                if let Some(counts) = &mut counts {
                    for point in &atlas {
                        if let Some(at) = registered.voxel(*point, Rounding::Nearest) {
                            let slot = &mut counts[(at[0] * shape[1] + at[1]) * shape[2] + at[2]];
                            *slot = slot.checked_add(1).ok_or_else(|| {
                                Error::invalid("tract: more than 2^32 points in one atlas voxel")
                            })?;
                        }
                    }
                }
                if let Some(out) = &mut csv {
                    let hemisphere = columns.hemisphere.as_deref();
                    for (row, (at, mapped)) in points.iter().zip(&atlas).enumerate() {
                        write!(
                            out,
                            "{},{},{},{},{},{},{}",
                            at[0],
                            at[1],
                            at[2],
                            mapped[0],
                            mapped[1],
                            mapped[2],
                            columns.region[row]
                        )
                        .map_err(Error::backend)?;
                        match hemisphere {
                            Some(hemisphere) => writeln!(out, ",{}", hemisphere[row]),
                            None => writeln!(out),
                        }
                        .map_err(Error::backend)?;
                    }
                }
            }
            Ok(())
        })?;
        eprintln!("tract: {mapped_count} points, {outside} outside the atlas");
        atlas_outside = Some(outside);
        if let (Some(path), Some(counts)) = (&args.atlas_density, counts) {
            let density = count_density(&counts, shape, params.density_radius)?;
            drop(counts);
            write_density(path, &density, atlas_lattice(registered), "micrometer")?;
        }
    } else if let Some(out) = &mut csv {
        let rescale = match &args.scale_to {
            Some(shape) => Some(Rescale::between(volume, triple(shape, "--scale-to")?)?),
            None => None,
        };
        writeln!(out, "z,y,x").map_err(Error::backend)?;
        for_each_point_slab(&env, &mut |slab| {
            for at in slab {
                let point = [at[0] as f64, at[1] as f64, at[2] as f64];
                let point = rescale.map_or(point, |rescale| rescale.apply(point));
                writeln!(out, "{},{},{}", point[0], point[1], point[2]).map_err(Error::backend)?;
            }
            Ok(())
        })?;
    }
    if let Some(out) = &mut csv {
        out.flush().map_err(Error::backend)?;
    }
    drop(csv);
    drop(env);

    let mask_label = format!("{}-mask", args.name);
    let density_label = format!("{}-density", args.name);
    let workers = args.input.workers();
    post.stage("publish labels");
    publish(&pyramid, &work, images.mask, &mask_label, workers)?;
    publish(&pyramid, &work, images.density, &density_label, workers)?;
    let _ = fs::remove_dir_all(&work);
    post.finish();
    Ok(TractSummary {
        range,
        points: point_count,
        mask_label,
        density_label,
        atlas_outside,
    })
}
