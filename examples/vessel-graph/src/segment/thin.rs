// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The skeleton's thinning to a fixed point over stored images, one round per
//! pixel phase, without holding the volume.
//!
//! [`super::skeleton::ThinToFixedPoint`] runs as one iterative phase, whose
//! two private buffers are the whole volume (two bytes a voxel). Here each
//! round is a pixel phase of its own ([`ThinRoundOp`], one directional pass
//! with the pass's own reach as halo), so a round reads the previous round's
//! stored image and writes its own, and a run of
//! [`ThinPlan::rounds_per_plan`] rounds is one plan. A round over a window is
//! the whole-volume pass on that window's core, because the halo is the pass's
//! exact reach; the rounds are those of the iterative phase, on any grid, and
//! the fixed point is the same image.
//!
//! **A block whose window did not change in the previous round does not
//! change in this one**: the pass is a function of the window, so this round's
//! core is the previous round's core. Each round also reads the image of the
//! round before the previous one over the same window, and where the two
//! windows are equal it copies instead of thinning. Most of the volume stops
//! changing within a few rounds; the last few hundred voxels take tens more.
//!
//! After each plan a fragment phase compares the last two images block by
//! block ([`ChangedOp`]); the thinning has converged when nothing changed in
//! the last round, the iterative phase's test.

use std::path::{Path, PathBuf};

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::env::BlockBuf;
use blockflow::fragment::{
    fold_fragments, BlockOutput, BlockView, Coverage, FragmentOp, FragmentOutput, SeamFold,
    SourceBlocks,
};
use blockflow::geometry::BlockGrid;
use blockflow::op::{Anchor, BlockOp, Chain, SourceInput, SourceInputs};
use blockflow::ops::directional::{
    directional_passes_into, directional_reach, DIRECTIONAL_PASS_COST,
};
use blockflow::reach::Reach;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::AttachedImage;
use blockflow::{Dtype, Error, Result, ZarrEnvironment};

use crate::profile::Listening;

const CHANGED_STREAM: &str = "segment.thin.changed";

/// One thinning round: `passes` directional passes over the block's window,
/// or a copy where the window is the one the previous round read.
pub struct ThinRoundOp {
    passes: usize,
    /// The image the previous round read (two rounds back), over the same
    /// window; `None` for the first round of all.
    earlier: Option<(ImageId, Option<Dtype>)>,
}

impl BlockOp for ThinRoundOp {
    fn name(&self) -> &'static str {
        "thinning round"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        directional_reach(self.passes)
    }

    fn accepts(&self, dtype: Dtype) -> bool {
        dtype == Dtype::Bool
    }

    fn produces(&self, _input: Dtype) -> Dtype {
        Dtype::Bool
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        self.earlier
            .iter()
            .map(|(image, dtype)| {
                let input =
                    SourceInput::new(*image, Reach::from([directional_reach(self.passes); 3]));
                match dtype {
                    Some(dtype) => input.holding(*dtype),
                    None => input,
                }
            })
            .collect()
    }

    fn apply(&self, input: &Voxels, out: &mut Voxels, _at: &Anchor) -> Result<()> {
        directional_passes_into(input.view::<bool>()?, self.passes, out.view_mut::<bool>()?)
    }

    fn apply_with(
        &self,
        input: &Voxels,
        sources: SourceInputs<'_>,
        out: &mut Voxels,
        at: &Anchor,
    ) -> Result<()> {
        let window = input.view::<bool>()?;
        if let Some((image, _)) = self.earlier {
            let earlier = sources.get(image)?;
            if earlier.shape() == input.shape() && earlier.view::<bool>()? == window {
                // The window the previous round read: its answer, which is the
                // core of this round's input.
                out.view_mut::<bool>()?.assign(&window);
                return Ok(());
            }
        }
        self.apply(input, out, at)
    }

    /// Nothing to thin in an empty block.
    fn constant_maps_to(&self, value: f64) -> Option<f64> {
        (value == 0.0).then_some(0.0)
    }

    fn cost_per_voxel(&self) -> f64 {
        DIRECTIONAL_PASS_COST * self.passes as f64
    }
}

/// Whether a block's core differs between its input and `earlier`: one byte
/// per block.
pub struct ChangedOp {
    earlier: ImageId,
}

impl FragmentOp for ChangedOp {
    fn name(&self) -> &'static str {
        "thinning changed"
    }

    fn reach(&self, _axis: usize, _volume_len: usize) -> usize {
        0
    }

    fn reads_pixels(&self) -> bool {
        true
    }

    fn writes_pixels(&self) -> bool {
        false
    }

    fn outputs(&self) -> Vec<FragmentOutput> {
        vec![FragmentOutput::new(
            CHANGED_STREAM,
            Lifecycle::DeleteOnExit,
            Coverage::EveryBlock,
        )]
    }

    fn source_inputs(&self, _volume: [usize; 3]) -> Vec<SourceInput> {
        vec![SourceInput::voxelwise(self.earlier)]
    }

    fn seam_fold(&self) -> Option<SeamFold> {
        Some(SeamFold::PerBlock)
    }

    fn apply(&self, _at: &BlockView<'_>) -> Result<BlockOutput> {
        Err(Error::invalid(
            "thinning changed compares two images and is applied through `apply_with`",
        ))
    }

    fn apply_with(&self, at: &BlockView<'_>, sources: SourceBlocks<'_>) -> Result<BlockOutput> {
        let (BlockBuf::Array(now), BlockBuf::Array(before)) =
            (at.pixels()?, sources.get(self.earlier.index())?)
        else {
            return Err(Error::invalid("thinning changed: a run without data"));
        };
        let changed = now.view::<bool>()? != before.view::<bool>()?;
        Ok(BlockOutput::fragment(
            CHANGED_STREAM,
            vec![u8::from(changed)],
        ))
    }
}

/// How the stored thinning runs. None of it changes the skeleton.
#[derive(Debug, Clone)]
pub struct ThinPlan {
    pub grid: [usize; 3],
    pub rounds_per_plan: usize,
    pub workers: usize,
    /// Rounds before the run is refused as not converging.
    pub round_limit: usize,
    pub passes: usize,
    /// Scratch directory, one subdirectory per plan; removed as it goes.
    pub work: PathBuf,
    /// The chunk cache of each plan's environment, in bytes; `None` for the
    /// environment's default, `Some(0)` for none.
    pub cache_bytes: Option<u64>,
}

/// The thinned image: a stored level-0 array, and the scratch directory that
/// holds it (the caller removes it once it has read the array).
#[derive(Debug, Clone)]
pub struct Thinned {
    pub level0: PathBuf,
    pub scratch: PathBuf,
    /// Rounds run, the last ones (up to a plan's worth) past the fixed point.
    pub rounds: usize,
}

fn level(work: &Path, image: ImageId) -> PathBuf {
    work.join(format!("level{}", image.index()))
}

/// Thin the stored `bool` image `mask` (a level-0 array) to its fixed point.
pub fn thin_stored(mask: &AttachedImage, plan: &ThinPlan) -> Result<Thinned> {
    if plan.passes == 0 || plan.rounds_per_plan == 0 || plan.round_limit == 0 {
        return Err(Error::invalid(
            "thinning: a round of zero passes, a plan of zero rounds or a limit of zero \
             rounds thins nothing",
        ));
    }
    let (dtype, volume) = mask.metadata()?;
    if dtype != Dtype::Bool {
        return Err(Error::invalid(format!(
            "thinning: the mask holds {}",
            dtype.numpy_name()
        )));
    }
    // The newest image and the one before it, with the directory each is in
    // (the first is the caller's and is not removed here).
    let mut newest = mask.clone();
    let mut newest_level: Option<PathBuf> = None;
    let mut previous: Option<AttachedImage> = None;
    let mut directories: Vec<PathBuf> = Vec::new();
    let mut rounds = 0usize;
    for run in 0.. {
        if rounds >= plan.round_limit {
            return Err(Error::invalid(format!(
                "thinning did not converge in {} rounds over a {volume:?} volume",
                plan.round_limit
            )));
        }
        let count = plan.rounds_per_plan.min(plan.round_limit - rounds).max(1);
        let mut builder = PlanBuilder::new(volume, Dtype::Bool, BlockGrid::new(volume, plan.grid)?);
        let mut attached = vec![newest.clone()];
        // `images[k]` is the image round `k` reads: image 0, then each round's.
        let mut images = vec![ImageId::from(0usize)];
        let mut earlier = previous.as_ref().map(|image| {
            attached.push(image.clone());
            (ImageId::supplied(0), Some(Dtype::Bool))
        });
        for round in 0..count {
            let written = builder
                .pixels(Chain::op(ThinRoundOp {
                    passes: plan.passes,
                    earlier,
                }))?
                .image()?;
            earlier = Some((images[round], None));
            images.push(written);
        }
        let last = images[count];
        let before = images[count - 1];
        builder.fragments(ChangedOp { earlier: before })?;
        let assembly = builder.finish()?;

        let work = plan.work.join(format!("plan{run}"));
        if work.exists() {
            std::fs::remove_dir_all(&work).map_err(Error::backend)?;
        }
        let mut env = ZarrEnvironment::attach(&work, &attached)?;
        match plan.cache_bytes {
            Some(0) => env = env.without_cache(),
            Some(bytes) => env = env.with_cache(bytes),
            None => {}
        }
        let listening = Listening::new("thinning", &assembly);
        execute_phases(
            "vessel-graph thinning",
            &assembly.workflow,
            &assembly.decomposition,
            &Hints {
                concurrency: plan.workers.max(1),
                keep_images: [last, before].into_iter().collect(),
                ..Hints::default()
            },
            &env,
            &listening.listeners(),
            &assembly.work(),
        )?;
        listening.finish();
        let mut changed = false;
        fold_fragments(&env, CHANGED_STREAM, &mut |_key, bytes| {
            changed |= bytes.first().copied().unwrap_or(1) != 0;
            Ok(())
        })?;
        drop(env);
        rounds += count;

        // The two newest images live on; every older directory goes.
        newest = AttachedImage::at(level(&work, last));
        newest_level = Some(level(&work, last));
        previous = Some(match before.index() {
            0 => attached[0].clone(),
            _ => AttachedImage::at(level(&work, before)),
        });
        directories.push(work);
        let keep = match before.index() {
            0 => 2,
            _ => 1,
        };
        while directories.len() > keep {
            let old = directories.remove(0);
            let _ = std::fs::remove_dir_all(old);
        }
        if !changed {
            break;
        }
    }
    // Only the newest directory is needed; the one before may hold the
    // second-newest image of a one-round plan.
    while directories.len() > 1 {
        let old = directories.remove(0);
        let _ = std::fs::remove_dir_all(old);
    }
    let scratch = directories.pop().unwrap_or_else(|| plan.work.clone());
    Ok(Thinned {
        level0: newest_level.unwrap_or_else(|| scratch.clone()),
        scratch,
        rounds,
    })
}
