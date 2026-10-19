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

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::geometry::BlockGrid;
use blockflow::ome_zarr::{publish_label, LabelPublication, OmeVolumePyramid};
use blockflow::op::Chain;
use blockflow::ops::WidenLabelsOp;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::{Error, Result, ZarrEnvironment};
use clap::Args;

use atlas_regions::Rounding;
use blockflow_atlas_registration::atlas_space::{LookupParams, OpenParams, RegisteredAtlas};

use crate::graph::cli::{atlas_lattice, image_frame, level_lattice, write_density};

use super::clip::ClipParams;
use super::crop::write_crop;
use super::params::{ChannelOverrides, ChannelRole, PostOverrides, SegmentOverrides};
use super::pipeline::{append_segment_phases, SegmentParams};
use super::plan::{workers, Cursor, Image};
use super::skeleton::SkeletonParams;
use super::tract::{
    append_clip_range_phases, append_tract_phases, count_points, read_clip_range, read_points,
    tract_binarize, Rescale, TractParams,
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
    /// Block edge, z,y,x.
    #[arg(long, value_delimiter = ',', default_values_t = vec![64, 256, 256])]
    pub block: Vec<usize>,
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

    fn block(&self) -> Result<[usize; 3]> {
        triple(&self.block, "--block")
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
            flags.apply(channel).map_err(|error| {
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

    let mut plan = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, args.input.block()?)?);
    let mut c = Cursor::new(&mut plan, sources[0]);
    let segmented = append_segment_phases(&mut c, &sources, &params)?;
    let mask = c.pixels(c.on(segmented.mask, vec![Chain::op(WidenLabelsOp)]))?;
    let skeleton = c.pixels(c.on(segmented.skeleton, vec![Chain::op(WidenLabelsOp)]))?;
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
    let channel_labels = match channel_masks.is_empty() {
        true => Vec::new(),
        false => channel_mask_labels(&args.name, &roles, &args.channel),
    };
    let assembly = plan.finish()?;

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
    execute_phases(
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
        &[],
        &assembly.work(),
    )?;
    drop(env);

    let mask_label = format!("{}-mask", args.name);
    let skeleton_label = format!("{}-skeleton", args.name);
    let workers = args.input.workers();
    publish(&pyramids[0], &work, mask, &mask_label, workers)?;
    publish(&pyramids[0], &work, skeleton, &skeleton_label, workers)?;
    for (image, label) in channel_masks.iter().zip(&channel_labels) {
        publish(&pyramids[0], &work, *image, label, workers)?;
    }
    let _ = fs::remove_dir_all(&work);
    eprintln!("segment: done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(SegmentSummary {
        mask_label,
        skeleton_label,
        channel_mask_labels: channel_labels,
        phases: assembly.n_phases(),
    })
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
    let grid = BlockGrid::new(volume, args.input.block()?)?;
    let source = Image::new(0usize, dtype);
    let work = args.input.work_dir("tract");

    // Run 1: the clip range.
    let mut plan = PlanBuilder::new(volume, dtype, grid.clone());
    append_clip_range_phases(&mut plan, source, &params)?;
    let assembly = plan.finish()?;
    fresh(&work)?;
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&attached))?;
    execute_phases(
        "vessel-graph tract range",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(&args.input, &[]),
        &env,
        &[],
        &assembly.work(),
    )?;
    let range = read_clip_range(&env)?;
    drop(env);
    eprintln!("tract: clip range {range:?}");

    // Run 2: mask, points, density.
    let mut plan = PlanBuilder::new(volume, dtype, grid);
    let mut c = Cursor::new(&mut plan, source);
    let images = append_tract_phases(&mut c, source, &params, range)?;
    let assembly = plan.finish()?;
    fresh(&work)?;
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&attached))?;
    execute_phases(
        "vessel-graph tract",
        &assembly.workflow,
        &assembly.decomposition,
        &hints(&args.input, &[images.mask.id, images.density.id]),
        &env,
        &[],
        &assembly.work(),
    )?;
    let points = match args.points_csv.is_some() || registered.is_some() {
        true => read_points(&env)?,
        false => Vec::new(),
    };
    let point_count = count_points(&env)?;
    drop(env);

    let mut atlas_outside = None;
    if let Some(registered) = &registered {
        let image = image_frame(pyramid.image_root(), level0);
        let to_sample = registered.image_to_sample(&image, [0.0; 3])?;
        if let Some(warning) = registered.frame_warning(&to_sample, volume) {
            eprintln!("tract: warning: {warning}");
        }
        let image_points: Vec<[f64; 3]> = points.iter().map(|p| p.map(|v| v as f64)).collect();
        let atlas = registered.map(&to_sample, &image_points).atlas;
        let lookup = LookupParams {
            level: args.level,
            rounding: Rounding::Nearest,
        };
        let columns = registered.lookup(&atlas, &lookup)?;
        let outside = registered.outside(&atlas, Rounding::Nearest);
        eprintln!("tract: {} points, {outside} outside the atlas", atlas.len());
        atlas_outside = Some(outside);
        if let Some(path) = &args.atlas_density {
            let density =
                registered.density(&atlas, None, Rounding::Nearest, params.density_radius)?;
            write_density(
                path,
                &density.voxels,
                atlas_lattice(registered),
                "micrometer",
            )?;
        }
        if let Some(path) = &args.points_csv {
            let mut out = std::io::BufWriter::new(fs::File::create(path).map_err(Error::backend)?);
            let hemispheres = columns.hemisphere.as_deref();
            write!(out, "z,y,x,atlas_z,atlas_y,atlas_x,annotation").map_err(Error::backend)?;
            writeln!(
                out,
                "{}",
                if hemispheres.is_some() {
                    ",hemisphere"
                } else {
                    ""
                }
            )
            .map_err(Error::backend)?;
            for (row, (at, mapped)) in points.iter().zip(&atlas).enumerate() {
                write!(
                    out,
                    "{},{},{},{},{},{},{}",
                    at[0], at[1], at[2], mapped[0], mapped[1], mapped[2], columns.region[row]
                )
                .map_err(Error::backend)?;
                match hemispheres {
                    Some(hemisphere) => writeln!(out, ",{}", hemisphere[row]),
                    None => writeln!(out),
                }
                .map_err(Error::backend)?;
            }
            out.flush().map_err(Error::backend)?;
        }
    } else if let Some(path) = &args.points_csv {
        let rescale = match &args.scale_to {
            Some(shape) => Some(Rescale::between(volume, triple(shape, "--scale-to")?)?),
            None => None,
        };
        let mut out = std::io::BufWriter::new(fs::File::create(path).map_err(Error::backend)?);
        writeln!(out, "z,y,x").map_err(Error::backend)?;
        for at in &points {
            let point = [at[0] as f64, at[1] as f64, at[2] as f64];
            let point = rescale.map_or(point, |rescale| rescale.apply(point));
            writeln!(out, "{},{},{}", point[0], point[1], point[2]).map_err(Error::backend)?;
        }
        out.flush().map_err(Error::backend)?;
    }

    let mask_label = format!("{}-mask", args.name);
    let density_label = format!("{}-density", args.name);
    let workers = args.input.workers();
    publish(&pyramid, &work, images.mask, &mask_label, workers)?;
    publish(&pyramid, &work, images.density, &density_label, workers)?;
    let _ = fs::remove_dir_all(&work);
    Ok(TractSummary {
        range,
        points: point_count,
        mask_label,
        density_label,
        atlas_outside,
    })
}
