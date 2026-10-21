// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The whole segmentation as one plan: every channel binarised, smoothed and
//! filled (and, where asked, deep-filled); the channels' masks united and
//! post-processed again; the result skeletonised.
//!
//! ```text
//! channel k:  raw -> binarize -> smooth -> fill [-> deep fill] --\
//!                                                                 +-> OR -> smooth -> fill -> mask
//! channel j:  raw -> binarize -> smooth -> fill [-> deep fill] --/                      \-> skeleton
//! ```
//!
//! With one channel there is nothing to unite and the channel's own filled
//! mask is the answer; the second smooth-and-fill runs only after a union,
//! where it closes the seams between channels.

use blockflow::assemble::{Assembly, ImageId, PlanBuilder};
use blockflow::geometry::BlockGrid;
use blockflow::log::Stats;
use blockflow::op::Chain;
use blockflow::ops::fill::append_connected;
use blockflow::ops::voxelwise::{Logic, LogicCombine};
use blockflow::ops::Connectivity;
use blockflow::sidecar::Lifecycle;
use blockflow::voxels::Voxels;
use blockflow::{Dtype, Error, Result};

use super::binarize::{append_binarize_phases, BinarizeParams, Binarized};
use super::deep_fill::{append_deep_fill, DeepFillOutput, DeepFillParams, DeepFillSetup};
use super::plan::{read_back, run_in_memory, Cursor, Image};
use super::skeleton::{clear_faces_node, SkeletonParams};
use super::smooth::SmoothParams;

/// What follows a binarisation, or a union: smoothing, then hole filling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostProcess {
    pub smooth: Option<SmoothParams>,
    /// Fill every background component that does not reach the volume's
    /// outside, connected as stated.
    pub fill: Option<Connectivity>,
}

impl Default for PostProcess {
    fn default() -> Self {
        Self {
            smooth: Some(SmoothParams::default()),
            fill: Some(Connectivity::Faces),
        }
    }
}

impl PostProcess {
    pub fn none() -> Self {
        Self {
            smooth: None,
            fill: None,
        }
    }
}

/// One channel: its binarisation and what follows it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChannelParams {
    pub binarize: BinarizeParams,
    pub post: PostProcess,
    /// The CNN lumen filling after the post-processing; `None` (the library
    /// default) skips it. The roles switch it on in a build with the
    /// `deep-fill` feature, as the reference's configuration does.
    pub deep_fill: Option<DeepFillParams>,
}

/// The whole segmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentParams {
    /// One entry per input channel, in input order.
    pub channels: Vec<ChannelParams>,
    /// After the union; used only with two channels or more.
    pub combined: PostProcess,
    pub skeleton: SkeletonParams,
    /// The weights and block grid of every channel's deep filling.
    pub deep_fill: DeepFillSetup,
}

impl SegmentParams {
    /// `channels` channels at the defaults.
    pub fn with_channels(channels: usize) -> Self {
        Self {
            channels: vec![ChannelParams::default(); channels],
            combined: PostProcess::default(),
            skeleton: SkeletonParams::default(),
            deep_fill: DeepFillSetup::default(),
        }
    }
}

/// The images one channel left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelImages {
    pub binarized: Binarized,
    /// After the channel's own smooth and fill, and deep fill where asked.
    pub processed: Image,
}

/// Where a segmentation plan's answers are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segmented {
    pub channels: Vec<ChannelImages>,
    /// The final mask, `Bool`.
    pub mask: Image,
    /// Its skeleton, `Bool`.
    pub skeleton: Image,
}

impl Segmented {
    /// Every image a caller may want to read back.
    pub fn images(&self) -> Vec<ImageId> {
        let mut images = vec![self.mask.id, self.skeleton.id];
        for channel in &self.channels {
            images.extend([
                channel.binarized.filtered.id,
                channel.binarized.binary.id,
                channel.processed.id,
            ]);
        }
        images
    }
}

/// Append smoothing and filling of `mask`.
pub fn append_post_process(
    c: &mut Cursor<'_>,
    mask: Image,
    post: &PostProcess,
    stream: &str,
) -> Result<Image> {
    let mut mask = mask;
    if let Some(smooth) = post.smooth {
        smooth.validate()?;
        mask = c.pixels(c.on(mask, vec![smooth.node()]))?;
    }
    if let Some(connectivity) = post.fill {
        // The labelling phase reads its input, so the mask has to be it.
        c.make_input(mask)?;
        let phase = append_connected(
            c.plan,
            stream,
            Lifecycle::DeleteOnExit,
            Dtype::Bool,
            connectivity,
        )?;
        mask = c.wrote(phase)?;
    }
    Ok(mask)
}

/// Append the face clearing and the thinning of `mask`.
pub fn append_skeleton(c: &mut Cursor<'_>, mask: Image, params: &SkeletonParams) -> Result<Image> {
    params.validate()?;
    if params.delete_border {
        c.pixels(c.on(mask, vec![clear_faces_node()]))?;
    } else {
        c.make_input(mask)?;
    }
    c.iterate(params.thinning()?)
}

/// The image the thinning starts from: `mask` with the volume's faces cleared
/// (if asked).
pub fn append_skeleton_input(
    c: &mut Cursor<'_>,
    mask: Image,
    params: &SkeletonParams,
) -> Result<Image> {
    params.validate()?;
    match params.delete_border {
        true => c.pixels(c.on(mask, vec![clear_faces_node()])),
        false => Ok(mask),
    }
}

/// Append the whole segmentation. `sources[k]` is channel `k`'s raw image; the
/// cursor's input should be image 0.
pub fn append_segment_phases(
    c: &mut Cursor<'_>,
    sources: &[Image],
    params: &SegmentParams,
) -> Result<Segmented> {
    append_segment(c, sources, params, true)
}

/// [`append_segment_phases`] without the thinning: [`Segmented::skeleton`] is
/// the image the thinning starts from ([`append_skeleton_input`]), for a
/// caller that thins over stored images ([`super::thin::thin_stored`]).
pub fn append_segment_phases_unthinned(
    c: &mut Cursor<'_>,
    sources: &[Image],
    params: &SegmentParams,
) -> Result<Segmented> {
    append_segment(c, sources, params, false)
}

fn append_segment(
    c: &mut Cursor<'_>,
    sources: &[Image],
    params: &SegmentParams,
    thin: bool,
) -> Result<Segmented> {
    if sources.is_empty() || sources.len() != params.channels.len() {
        return Err(Error::invalid(format!(
            "segment: {} source channel(s) and parameters for {}",
            sources.len(),
            params.channels.len()
        )));
    }
    let mut channels = Vec::with_capacity(sources.len());
    for (index, (raw, channel)) in sources.iter().zip(&params.channels).enumerate() {
        let binarized = append_binarize_phases(c, *raw, &channel.binarize)?;
        let mut processed = append_post_process(
            c,
            binarized.binary,
            &channel.post,
            &format!("segment.channel{index}.background"),
        )?;
        if let Some(deep_fill) = &channel.deep_fill {
            processed = append_deep_fill(
                c,
                processed,
                deep_fill,
                &params.deep_fill,
                DeepFillOutput::Filled,
            )?;
        }
        channels.push(ChannelImages {
            binarized,
            processed,
        });
    }
    let mask = if channels.len() == 1 {
        channels[0].processed
    } else {
        let union = c.pixels(Chain::parallel(
            channels
                .iter()
                .map(|channel| c.on(channel.processed, vec![]))
                .collect(),
            Box::new(LogicCombine::new("union", Logic::Or).producing(Dtype::Bool)?),
        )?)?;
        append_post_process(c, union, &params.combined, "segment.combined.background")?
    };
    let skeleton = match thin {
        true => append_skeleton(c, mask, &params.skeleton)?,
        false => append_skeleton_input(c, mask, &params.skeleton)?,
    };
    Ok(Segmented {
        channels,
        mask,
        skeleton,
    })
}

/// The arrays of one channel after an in-memory run.
#[derive(Debug, Clone)]
pub struct ChannelOutput {
    pub filtered: Voxels,
    pub binary: Voxels,
    pub processed: Voxels,
}

/// An in-memory segmentation's answers.
#[derive(Debug)]
pub struct Segmentation {
    pub channels: Vec<ChannelOutput>,
    pub mask: Voxels,
    pub skeleton: Voxels,
    pub stats: Stats,
}

/// The segmentation plan over channels of `dtypes` (image 0, then the
/// supplied images), cut into `block`, without running it.
pub fn segment_plan(
    volume: [usize; 3],
    dtypes: &[Dtype],
    params: &SegmentParams,
    block: [usize; 3],
) -> Result<(Assembly, Segmented)> {
    let Some(&first) = dtypes.first() else {
        return Err(Error::invalid("segment: no channels"));
    };
    let sources: Vec<Image> = dtypes
        .iter()
        .enumerate()
        .map(|(index, &dtype)| match index {
            0 => Image::new(0usize, dtype),
            _ => Image::new(ImageId::supplied(index - 1), dtype),
        })
        .collect();
    let mut plan = PlanBuilder::new(volume, first, BlockGrid::new(volume, block)?);
    let mut c = Cursor::new(&mut plan, sources[0]);
    let segmented = append_segment_phases(&mut c, &sources, params)?;
    Ok((plan.finish()?, segmented))
}

/// Segment channels held in memory (all of one shape), cut into `block`.
pub fn segment(
    volumes: Vec<Voxels>,
    params: &SegmentParams,
    block: [usize; 3],
) -> Result<Segmentation> {
    let Some(first) = volumes.first() else {
        return Err(Error::invalid("segment: no channels"));
    };
    let volume = first.shape();
    if let Some(other) = volumes.iter().find(|other| other.shape() != volume) {
        return Err(Error::ShapeMismatch {
            expected: volume.to_vec(),
            got: other.shape().to_vec(),
        });
    }
    let dtypes: Vec<Dtype> = volumes.iter().map(Voxels::dtype).collect();
    let (assembly, segmented) = segment_plan(volume, &dtypes, params, block)?;
    let (stats, env) = run_in_memory(&assembly, volumes, &segmented.images())?;
    let channels = segmented
        .channels
        .iter()
        .map(|channel| {
            Ok(ChannelOutput {
                filtered: read_back(&env, channel.binarized.filtered)?,
                binary: read_back(&env, channel.binarized.binary)?,
                processed: read_back(&env, channel.processed)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Segmentation {
        channels,
        mask: read_back(&env, segmented.mask)?,
        skeleton: read_back(&env, segmented.skeleton)?,
        stats,
    })
}

/// One channel's binarisation plan, without running it.
pub fn binarize_plan(
    volume: [usize; 3],
    dtype: Dtype,
    params: &BinarizeParams,
    block: [usize; 3],
) -> Result<(Assembly, Binarized)> {
    let source = Image::new(0usize, dtype);
    let mut plan = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, block)?);
    let mut c = Cursor::new(&mut plan, source);
    let binarized = append_binarize_phases(&mut c, source, params)?;
    Ok((plan.finish()?, binarized))
}

/// One channel's binarisation alone, in memory: the filtered volume and the
/// sink.
pub fn binarize(
    volume: Voxels,
    params: &BinarizeParams,
    block: [usize; 3],
) -> Result<(Voxels, Voxels)> {
    let (assembly, binarized) = binarize_plan(volume.shape(), volume.dtype(), params, block)?;
    let (_, env) = run_in_memory(
        &assembly,
        vec![volume],
        &[binarized.filtered.id, binarized.binary.id],
    )?;
    Ok((
        read_back(&env, binarized.filtered)?,
        read_back(&env, binarized.binary)?,
    ))
}

/// What a mask-only run is asked to do: [`mask_steps`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaskSteps {
    pub post: Option<PostProcess>,
    pub skeleton: Option<SkeletonParams>,
}

/// Post-process and/or skeletonise a `Bool` mask held in memory. Returns the
/// run's statistics, the post-processed mask and the skeleton (when asked).
pub fn mask_steps(
    mask: Voxels,
    steps: &MaskSteps,
    block: [usize; 3],
) -> Result<(Stats, Voxels, Option<Voxels>)> {
    if mask.dtype() != Dtype::Bool {
        return Err(Error::invalid(format!(
            "mask steps: the input holds {} and these steps take a mask",
            mask.dtype().numpy_name()
        )));
    }
    let shape = mask.shape();
    let source = Image::new(0usize, Dtype::Bool);
    let mut plan = PlanBuilder::new(shape, Dtype::Bool, BlockGrid::new(shape, block)?);
    let mut c = Cursor::new(&mut plan, source);
    let processed = match &steps.post {
        Some(post) => append_post_process(&mut c, source, post, "mask.background")?,
        None => source,
    };
    let skeleton = match &steps.skeleton {
        Some(params) => Some(append_skeleton(&mut c, processed, params)?),
        None => None,
    };
    if c.plan.n_phases() == 0 {
        return Err(Error::invalid("mask steps: nothing was asked for"));
    }
    let assembly = plan.finish()?;
    let mut keep = vec![processed.id];
    keep.extend(skeleton.map(|image| image.id));
    let (stats, env) = run_in_memory(&assembly, vec![mask], &keep)?;
    let processed = read_back(&env, processed)?;
    let skeleton = match skeleton {
        Some(image) => Some(read_back(&env, image)?),
        None => None,
    };
    Ok((stats, processed, skeleton))
}
