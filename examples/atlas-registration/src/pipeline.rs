// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The two commands: register a moving OME-Zarr volume to a fixed one, and
//! carry a point table from the fixed volume's voxels to the moving volume's.
//!
//! **Direction.** The transform maps fixed coordinates to moving coordinates,
//! as elastix's does. A point found in the fixed volume can be pushed through
//! it; a point found in the moving volume cannot. To carry detections from a
//! sample into an atlas, register with the sample as `--fixed` and the atlas as
//! `--moving`; the warped output is then the atlas resampled into the sample.

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::error::{Error, Result};
use blockflow::ops::TransformInterpolation;
use blockflow::voxels::Voxels;
use blockflow::Dtype;
use ndarray::Array3;
use serde_json::{json, Value};

use crate::compute::{
    point_transform, transform_text, with_initial, FitConvergence, SpatialFrame, TransformModel,
    VolumeFitParams,
};
use crate::rescale::{Frame, Rescale};
use crate::similarity::{similarity, Similarity};
use crate::stochastic::{
    fit_stochastic, AffineMap, StochasticFitParams, StochasticInitialisation, StochasticLevelReport,
};
use crate::volume::{lattice_over, warp, write_image, Volume};
use crate::{AtlasTransform, ElastixTransform, PointSpace, RegisterParams};

/// The sidecar `register` writes beside the transform files.
pub const REGISTRATION_JSON: &str = "registration.json";

/// One side of a registration: which image, channel and pyramid level.
#[derive(Debug, Clone)]
pub struct Input {
    pub path: PathBuf,
    pub channel: usize,
    /// `None`: the coarsest level no coarser than the lattice.
    pub level: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RegisterConfig {
    pub fixed: Input,
    pub moving: Input,
    /// The registration lattice's voxel size, `[z, y, x]`, in the images'
    /// physical unit. `None`: the fixed image's level, as stored.
    pub resolution: Option<[f64; 3]>,
    /// Fit stages in order, each composed onto the previous ones — an affine
    /// then a B-spline, typically. All native or all stochastic.
    pub stages: Vec<Stage>,
    /// Directory for `TransformParameters.N.txt` and `registration.json`.
    pub out: PathBuf,
    /// Write the warped volume here as OME-Zarr.
    pub warped: Option<PathBuf>,
    /// Warp this image instead of the moving one: an annotation on the
    /// moving image's physical frame, for example.
    pub warp_source: Option<Input>,
    /// Nearest-neighbour warp, for labels.
    pub nearest: bool,
    pub block: usize,
    pub workers: usize,
    /// Also write the moving image warped after each stage, as
    /// `stage{N}_warped.ome.zarr` in `out` (stochastic stages).
    pub warped_stages: bool,
    /// How the first stage starts (stochastic stages).
    pub initialisation: Initialisation,
}

/// How a registration's first stage starts: where the moving image is taken
/// to lie against the fixed one before the fit moves it. Later stages start
/// from the stages before them.
///
/// The elastix equivalents: `Parameters` follows the file's
/// `AutomaticTransformInitialization`; `Physical` is that setting `"false"`
/// (the images' physical frames as they stand, elastix's identity); `Centre`
/// is `"true"` with `AutomaticTransformInitializationMethod
/// "GeometricalCenter"`; `Translation` is the physical frames with an initial
/// translation, as elastix's `-t0` with a translation file, except that the
/// translation is the first stage's starting value rather than a separate
/// transform (so it is fitted with it, and written into its file).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Initialisation {
    #[default]
    Parameters,
    Physical,
    Centre,
    /// `[z, y, x]` in the images' physical unit: the fixed image's point `p`
    /// starts at the moving image's point `p + t`.
    Translation([f64; 3]),
}

impl Initialisation {
    /// `parameters`, `physical`, `centre` (or `center`).
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "parameters" | "file" => Ok(Self::Parameters),
            "physical" | "origins" => Ok(Self::Physical),
            "centre" | "center" | "geometricalcenter" => Ok(Self::Centre),
            other => Err(Error::invalid(format!(
                "initialisation `{other}`: parameters, physical or centre"
            ))),
        }
    }

    /// The sidecar's record of it.
    pub fn to_json(self) -> Value {
        match self {
            Self::Parameters => json!("parameters"),
            Self::Physical => json!("physical"),
            Self::Centre => json!("centre"),
            Self::Translation(t) => json!({ "translation": t }),
        }
    }

    /// Apply to the first stochastic stage on a lattice of `spacing`: the
    /// stage's initialisation, and the explicit starting parameters (fixed
    /// lattice voxels, `ops::align`'s internal layout) for a translation.
    fn apply(
        self,
        params: &mut StochasticFitParams,
        spacing: [f64; 3],
    ) -> Result<Option<Vec<f64>>> {
        match self {
            Self::Parameters => Ok(None),
            Self::Physical => {
                params.initialisation = StochasticInitialisation::Identity;
                Ok(None)
            }
            Self::Centre => {
                params.initialisation = StochasticInitialisation::GeometricCentre;
                Ok(None)
            }
            Self::Translation(t) => {
                params.initialisation = StochasticInitialisation::Identity;
                let voxels: [f64; 3] = std::array::from_fn(|a| t[a] / spacing[a]);
                match params.model {
                    TransformModel::Translation => Ok(Some(voxels.to_vec())),
                    TransformModel::Affine => {
                        let mut values = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
                        values.extend(voxels);
                        Ok(Some(values))
                    }
                    TransformModel::BSpline => Err(Error::invalid(
                        "an initial translation needs a translation or affine first stage, not a \
                         B-spline",
                    )),
                }
            }
        }
    }
}

/// One fit stage.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// `ops::align`'s block-reduced Gauss-Newton fit. A later native stage fits
    /// the moving image as warped by the earlier ones.
    Native(VolumeFitParams),
    /// `ops::align::fit_stochastic`: Mattes mutual information with elastix's
    /// stochastic gradient descent. A later stage is fitted after the earlier
    /// ones (elastix's `Compose`): a B-spline after an affine lives in the
    /// space the affine maps into.
    Stochastic(StochasticFitParams),
}

impl From<VolumeFitParams> for Stage {
    fn from(params: VolumeFitParams) -> Self {
        Self::Native(params)
    }
}

impl From<StochasticFitParams> for Stage {
    fn from(params: StochasticFitParams) -> Self {
        Self::Stochastic(params)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageSummary {
    pub model: TransformModel,
    /// `None` for a coarse-to-fine stage, which reports no diagnostics.
    pub substages: Option<usize>,
    pub final_cost: Option<f64>,
    pub converged_by: Option<FitConvergence>,
    /// Per level, for a stochastic stage.
    pub levels: Vec<StochasticLevelReport>,
    pub seconds: Option<f64>,
    /// Fixed lattice against the moving image warped through every stage up
    /// to this one, for a stochastic stage.
    pub similarity: Option<Similarity>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegisterSummary {
    pub lattice_shape: [usize; 3],
    pub lattice_frame: Frame,
    pub fixed_level: usize,
    pub moving_level: usize,
    pub stages: Vec<StageSummary>,
    /// The outermost file: what `transformix -tp` would be handed.
    pub transform_file: PathBuf,
    pub warped: Option<PathBuf>,
}

pub fn register(config: &RegisterConfig) -> Result<RegisterSummary> {
    if config.stages.is_empty() {
        return Err(Error::invalid("register: at least one fit stage is needed"));
    }
    let native: Vec<VolumeFitParams> = config
        .stages
        .iter()
        .filter_map(|stage| match stage {
            Stage::Native(params) => Some(params.clone()),
            Stage::Stochastic(_) => None,
        })
        .collect();
    let stochastic: Vec<StochasticFitParams> = config
        .stages
        .iter()
        .filter_map(|stage| match stage {
            Stage::Stochastic(params) => Some(params.clone()),
            Stage::Native(_) => None,
        })
        .collect();
    match (native.is_empty(), stochastic.is_empty()) {
        (false, true) if config.initialisation != Initialisation::Parameters => Err(Error::invalid(
            "register: an initialisation other than the parameters' applies to stochastic stages \
             (elastix parameter files with a stochastic optimiser) only",
        )),
        (false, true) => register_native(config, &native),
        (true, false) => register_stochastic(config, &stochastic),
        _ => Err(Error::invalid(
            "register: native (Gauss-Newton) and stochastic stages compose differently and \
             cannot be mixed in one run",
        )),
    }
}

fn register_native(config: &RegisterConfig, native: &[VolumeFitParams]) -> Result<RegisterSummary> {
    let fixed = Volume::open(&config.fixed.path, config.fixed.channel)?;
    let moving = Volume::open(&config.moving.path, config.moving.channel)?;

    // The lattice: the fixed image's box at the requested voxel size.
    let fixed_level = match (config.fixed.level, config.resolution) {
        (Some(level), _) => level,
        (None, Some(spacing)) => fixed.level_for(spacing),
        (None, None) => 0,
    };
    check_level(&fixed, fixed_level, "fixed")?;
    let (lattice, lattice_shape) = match config.resolution {
        Some(spacing) => {
            lattice_over(&fixed.frame(fixed_level), fixed.shape(fixed_level), spacing)?
        }
        None => (fixed.frame(fixed_level), fixed.shape(fixed_level)),
    };
    let moving_level = config
        .moving
        .level
        .unwrap_or_else(|| moving.level_for(lattice.scale));
    check_level(&moving, moving_level, "moving")?;

    fs::create_dir_all(&config.out).map_err(Error::backend)?;
    let work = config.out.join("work");
    let sample = |volume: &Volume, level: usize, name: &str| -> Result<Array3<f64>> {
        Ok(volume
            .sample_onto(
                level,
                &lattice,
                lattice_shape,
                TransformInterpolation::Linear,
                &work.join(name),
                config.block,
            )?
            .widened())
    };
    // Unit peak intensity: the optimiser's tolerance is absolute, so the
    // images' own scale (a 16-bit stack, an 8-bit atlas) would otherwise
    // decide when it stops.
    let fixed_lattice = Voxels::from(unit_peak(sample(&fixed, fixed_level, "fixed")?));
    let moving_lattice = unit_peak(sample(&moving, moving_level, "moving")?);
    // Each sample removes its own run directory; this is their empty parent.
    let _ = fs::remove_dir(&work);

    let frame = SpatialFrame::new(lattice.translation, lattice.scale, IDENTITY)?;
    let mut chain: Vec<ElastixTransform> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut stages = Vec::new();
    for params in native {
        let params = params.clone().with_geometry(frame.clone());
        // Each stage fits the moving image as warped by the stages before it,
        // resampled once from the original rather than stage by stage.
        let current = if chain.is_empty() {
            Voxels::from(moving_lattice.clone())
        } else {
            let composed = |p: [f64; 3]| apply_chain(&chain, p);
            Voxels::from(warp(
                &moving_lattice,
                &lattice,
                &lattice,
                lattice_shape,
                &composed,
                TransformInterpolation::Linear,
            )?)
        };
        let (fitted, summary) = if params.pyramid.is_some() {
            let fitted = blockflow::ops::align::resident(
                &params,
                &fixed_lattice.view::<f64>()?.to_owned(),
                &current.view::<f64>()?.to_owned(),
            )?;
            let summary = StageSummary {
                model: params.model,
                substages: None,
                final_cost: None,
                converged_by: None,
                levels: Vec::new(),
                seconds: None,
                similarity: None,
            };
            (fitted, summary)
        } else {
            let diagnostics = blockflow::ops::align::run_with_diagnostics(
                &params,
                &fixed_lattice,
                &current,
                config.block,
                config.workers,
            )?;
            let summary = StageSummary {
                model: params.model,
                substages: Some(diagnostics.substages),
                final_cost: Some(diagnostics.final_cost),
                converged_by: Some(diagnostics.converged_by),
                levels: Vec::new(),
                seconds: None,
                similarity: None,
            };
            (diagnostics.transform, summary)
        };
        // A later stage is applied *first*: the whole map is
        // `T_1(T_2(… T_k(x)))`, so stage k is the innermost link.
        chain.insert(0, point_transform(&fitted)?);
        texts.insert(0, transform_text(&fitted)?);
        stages.push(summary);
    }

    // Innermost first on disk, each naming the one before it.
    let mut transform_file = PathBuf::new();
    for (index, text) in texts.iter().enumerate() {
        let text = if index == 0 {
            text.clone()
        } else {
            with_initial(text, &format!("TransformParameters.{}.txt", index - 1))
        };
        transform_file = config.out.join(format!("TransformParameters.{index}.txt"));
        fs::write(&transform_file, text).map_err(Error::backend)?;
    }

    let warped = match &config.warped {
        Some(path) => {
            let composed = |p: [f64; 3]| apply_chain(&chain, p);
            write_warped(
                config,
                &moving,
                moving_level,
                &lattice,
                lattice_shape,
                &composed,
                path,
            )?;
            Some(path.clone())
        }
        None => None,
    };

    let sidecar = json!({
        "schema": 1,
        "axis_order": ["z", "y", "x"],
        "transform": transform_file.file_name().and_then(|n| n.to_str()),
        "direction": "fixed physical -> moving physical",
        "fixed": side_json(&config.fixed, &fixed, fixed_level),
        "moving": side_json(&config.moving, &moving, moving_level),
        "lattice": { "shape": lattice_shape, "frame": frame_json(&lattice) },
        "optimizer": "gauss-newton",
        "stages": stages.iter().map(stage_json).collect::<Vec<_>>(),
        "warped": warped.as_ref().map(|path| path.display().to_string()),
    });
    fs::write(
        config.out.join(REGISTRATION_JSON),
        serde_json::to_vec_pretty(&sidecar).map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;

    Ok(RegisterSummary {
        lattice_shape,
        lattice_frame: lattice,
        fixed_level,
        moving_level,
        stages,
        transform_file,
        warped,
    })
}

fn stage_json(stage: &StageSummary) -> Value {
    json!({
        "model": format!("{:?}", stage.model),
        "substages": stage.substages,
        "final_cost": stage.final_cost,
        "converged_by": stage.converged_by.map(|c| format!("{c:?}")),
        "seconds": stage.seconds,
        "similarity": stage.similarity.map(|s| json!({
            "correlation": s.correlation,
            "mutual_information": s.mutual_information,
        })),
        "levels": stage.levels.iter().map(|level| json!({
            "iterations": level.iterations,
            "initial_cost": level.initial_cost,
            "final_cost": level.final_cost,
            "gain": { "a": level.gain.a, "A": level.gain.big_a, "alpha": level.gain.alpha },
            "stopped_by_tolerance": level.stopped_by_tolerance,
            "seconds": level.seconds,
            "grid": level.grid,
        })).collect::<Vec<_>>(),
    })
}

/// The voxel map from one lattice to another through physical coordinates:
/// `index_to = (scale_from * index_from + translation_from - translation_to) /
/// scale_to`.
fn lattice_map(from: &Frame, to: &Frame) -> AffineMap {
    let mut map = AffineMap::identity();
    for a in 0..3 {
        map.matrix[a][a] = from.scale[a] / to.scale[a];
        map.offset[a] = (from.translation[a] - to.translation[a]) / to.scale[a];
    }
    map
}

/// The stochastic registration: both images sampled onto lattices of the same
/// voxel size, each over its own box, and fitted by
/// `ops::align::fit_stochastic` stage after stage, each stage after the ones
/// before it, as elastix chains its parameter files.
fn register_stochastic(
    config: &RegisterConfig,
    stages: &[StochasticFitParams],
) -> Result<RegisterSummary> {
    let fixed = Volume::open(&config.fixed.path, config.fixed.channel)?;
    let moving = Volume::open(&config.moving.path, config.moving.channel)?;
    let fixed_level = match (config.fixed.level, config.resolution) {
        (Some(level), _) => level,
        (None, Some(spacing)) => fixed.level_for(spacing),
        (None, None) => 0,
    };
    check_level(&fixed, fixed_level, "fixed")?;
    let spacing = config
        .resolution
        .unwrap_or_else(|| fixed.frame(fixed_level).scale);
    let (lattice, lattice_shape) =
        lattice_over(&fixed.frame(fixed_level), fixed.shape(fixed_level), spacing)?;
    let moving_level = config
        .moving
        .level
        .unwrap_or_else(|| moving.level_for(spacing));
    check_level(&moving, moving_level, "moving")?;
    let (moving_lattice, moving_shape) = lattice_over(
        &moving.frame(moving_level),
        moving.shape(moving_level),
        spacing,
    )?;

    fs::create_dir_all(&config.out).map_err(Error::backend)?;
    let mut timings = serde_json::Map::new();
    let mut clock = std::time::Instant::now();
    let mut lap = |name: &str, timings: &mut serde_json::Map<String, Value>| {
        timings.insert(name.to_string(), json!(clock.elapsed().as_secs_f64()));
        clock = std::time::Instant::now();
    };
    // At registration resolution both images fit in memory, so the level is
    // read whole and resampled resident (in parallel slabs); a lattice that is
    // the level itself is used as read.
    let sample = |volume: &Volume, level: usize, frame: &Frame, shape: [usize; 3]| {
        let source = volume.read(level)?.widened();
        if volume.frame(level) == *frame && volume.shape(level) == shape {
            return Ok(source);
        }
        let identity = |p: [f64; 3]| p;
        warp(
            &source,
            &volume.frame(level),
            frame,
            shape,
            &identity,
            TransformInterpolation::Linear,
        )
    };
    let fixed_lattice = sample(&fixed, fixed_level, &lattice, lattice_shape)?;
    let moving_lattice_image = sample(&moving, moving_level, &moving_lattice, moving_shape)?;
    let fixed_f32 = fixed_lattice.mapv(|v| v as f32);
    let moving_f32 = moving_lattice_image.mapv(|v| v as f32);
    lap("sample_s", &mut timings);

    let frame = SpatialFrame::new(lattice.translation, lattice.scale, IDENTITY)?;
    let prior = lattice_map(&lattice, &moving_lattice);
    let mut pre = AffineMap::identity();
    let mut after_bspline = false;
    let mut summaries = Vec::new();
    let mut transform_file = PathBuf::new();
    for (index, params) in stages.iter().enumerate() {
        if after_bspline {
            return Err(Error::invalid(
                "register: a stage after a B-spline stage is not supported (the B-spline would \
                 have to be part of the next stage's fixed map)",
            ));
        }
        let mut params = params.clone();
        if params.workers == 0 {
            params.workers = config.workers;
        }
        let initial = if index == 0 {
            config.initialisation.apply(&mut params, lattice.scale)?
        } else {
            None
        };
        let fit = fit_stochastic(
            &params,
            fixed_f32.view(),
            moving_f32.view(),
            &pre,
            &prior,
            initial,
            frame.clone(),
        )?;
        lap(&format!("stage{index}_fit_s"), &mut timings);
        let text = transform_text(&fit.transform)?;
        let text = if index == 0 {
            text
        } else {
            with_initial(&text, &format!("TransformParameters.{}.txt", index - 1))
        };
        transform_file = config.out.join(format!("TransformParameters.{index}.txt"));
        fs::write(&transform_file, text).map_err(Error::backend)?;
        match fit.transform.model {
            TransformModel::Translation => {
                let t = &fit.transform.params;
                let shift = AffineMap {
                    matrix: IDENTITY,
                    offset: [t[0], t[1], t[2]],
                };
                pre = shift.then_after(&pre);
            }
            TransformModel::Affine => {
                pre = AffineMap::from_params(&fit.transform.params)?.then_after(&pre);
            }
            TransformModel::BSpline => after_bspline = true,
        }
        // How well the stages so far fit, on the fixed lattice.
        let so_far = ElastixTransform::read(&transform_file)?;
        let map = |p: [f64; 3]| so_far.apply(p, &RegisterParams::default());
        let warped = warp(
            &moving_lattice_image,
            &moving_lattice,
            &lattice,
            lattice_shape,
            &map,
            TransformInterpolation::Linear,
        )?;
        let measured = similarity(&fixed_lattice, &warped);
        if config.warped_stages {
            write_image(
                &config.out.join(format!("stage{index}_warped.ome.zarr")),
                &Voxels::from(warped.mapv(|v| v as f32)),
                &lattice,
                moving.unit().as_deref(),
            )?;
        }
        lap(&format!("stage{index}_similarity_s"), &mut timings);
        summaries.push(StageSummary {
            model: fit.transform.model,
            substages: Some(fit.levels.iter().map(|l| l.iterations).sum()),
            final_cost: fit.levels.last().map(|l| l.final_cost),
            converged_by: None,
            levels: fit.levels,
            seconds: Some(fit.seconds),
            similarity: Some(measured),
        });
    }

    let warped = match &config.warped {
        Some(path) => {
            let whole = ElastixTransform::read(&transform_file)?;
            let map = |p: [f64; 3]| whole.apply(p, &RegisterParams::default());
            write_warped(
                config,
                &moving,
                moving_level,
                &lattice,
                lattice_shape,
                &map,
                path,
            )?;
            Some(path.clone())
        }
        None => None,
    };
    lap("warp_s", &mut timings);
    let sidecar = json!({
        "schema": 1,
        "axis_order": ["z", "y", "x"],
        "transform": transform_file.file_name().and_then(|n| n.to_str()),
        "direction": "fixed physical -> moving physical",
        "optimizer": "stochastic",
        "initialisation": config.initialisation.to_json(),
        "timings": timings,
        "fixed": side_json(&config.fixed, &fixed, fixed_level),
        "moving": side_json(&config.moving, &moving, moving_level),
        "lattice": { "shape": lattice_shape, "frame": frame_json(&lattice) },
        "moving_lattice": { "shape": moving_shape, "frame": frame_json(&moving_lattice) },
        "stages": summaries.iter().map(stage_json).collect::<Vec<_>>(),
        "warped": warped.as_ref().map(|path| path.display().to_string()),
    });
    fs::write(
        config.out.join(REGISTRATION_JSON),
        serde_json::to_vec_pretty(&sidecar).map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;
    Ok(RegisterSummary {
        lattice_shape,
        lattice_frame: lattice,
        fixed_level,
        moving_level,
        stages: summaries,
        transform_file,
        warped,
    })
}

// -------------------------------------------------------------- chains --

/// The sidecar `chain` writes: the registrations of a chain, in order.
pub const CHAIN_JSON: &str = "chain.json";

/// A chain of registrations over a list of images: image `k` is registered
/// (fixed) to image `k + 1` (moving), so points go from the first image to
/// the last. The reference's cell pipeline is the two-step chain signal →
/// autofluorescence → atlas.
#[derive(Debug, Clone)]
pub struct ChainConfig {
    pub images: Vec<Input>,
    /// The stages of each step, `images.len() - 1` of them.
    pub steps: Vec<Vec<Stage>>,
    pub resolution: Option<[f64; 3]>,
    pub out: PathBuf,
    pub block: usize,
    pub workers: usize,
    pub warped_stages: bool,
    /// Write each step's moving image warped onto its fixed lattice.
    pub warped: bool,
    /// Per step, how its first stage starts; a step beyond the list follows
    /// its parameters.
    pub initialisation: Vec<Initialisation>,
}

/// Run every step of a chain into `out/step{k}/` and write `out/chain.json`.
pub fn register_chain(config: &ChainConfig) -> Result<Vec<RegisterSummary>> {
    if config.images.len() < 2 || config.steps.len() + 1 != config.images.len() {
        return Err(Error::invalid(format!(
            "chain: {} image(s) need {} step(s) of stages, got {}",
            config.images.len(),
            config.images.len().saturating_sub(1),
            config.steps.len()
        )));
    }
    fs::create_dir_all(&config.out).map_err(Error::backend)?;
    let mut summaries = Vec::new();
    let mut names = Vec::new();
    for (index, stages) in config.steps.iter().enumerate() {
        let name = format!("step{index}");
        let out = config.out.join(&name);
        let summary = register(&RegisterConfig {
            fixed: config.images[index].clone(),
            moving: config.images[index + 1].clone(),
            resolution: config.resolution,
            stages: stages.clone(),
            out: out.clone(),
            warped: config.warped.then(|| out.join("warped.ome.zarr")),
            warp_source: None,
            nearest: false,
            block: config.block,
            workers: config.workers,
            warped_stages: config.warped_stages,
            initialisation: config
                .initialisation
                .get(index)
                .copied()
                .unwrap_or_default(),
        })?;
        summaries.push(summary);
        names.push(name);
    }
    let sidecar = json!({
        "schema": 1,
        "steps": names,
        "images": config.images.iter().map(|i| i.path.display().to_string()).collect::<Vec<_>>(),
        "direction": "first image level-0 voxels -> last image level-0 voxels, [z, y, x]",
    });
    fs::write(
        config.out.join(CHAIN_JSON),
        serde_json::to_vec_pretty(&sidecar).map_err(Error::backend)?,
    )
    .map_err(Error::backend)?;
    Ok(summaries)
}

fn unit_peak(mut image: Array3<f64>) -> Array3<f64> {
    let peak = image.iter().copied().fold(0.0, f64::max);
    if peak > 0.0 {
        image.mapv_inplace(|v| v / peak);
    }
    image
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn check_level(volume: &Volume, level: usize, name: &str) -> Result<()> {
    if level >= volume.levels() {
        return Err(Error::invalid(format!(
            "register: the {name} image has {} level(s), so level {level} does not exist",
            volume.levels()
        )));
    }
    Ok(())
}

fn apply_chain(chain: &[ElastixTransform], point: [f64; 3]) -> [f64; 3] {
    let params = RegisterParams::default();
    chain
        .iter()
        .rev()
        .fold(point, |at, transform| transform.apply(at, &params))
}

fn write_warped(
    config: &RegisterConfig,
    moving: &Volume,
    moving_level: usize,
    lattice: &Frame,
    lattice_shape: [usize; 3],
    map: &(dyn Fn([f64; 3]) -> [f64; 3] + Sync),
    path: &Path,
) -> Result<()> {
    let opened;
    let (source, level) = match &config.warp_source {
        Some(input) => {
            opened = Volume::open(&input.path, input.channel)?;
            let level = input
                .level
                .unwrap_or_else(|| opened.level_for(lattice.scale));
            check_level(&opened, level, "warp source")?;
            (&opened, level)
        }
        None => (moving, moving_level),
    };
    let stored = source.read(level)?;
    let integer = !matches!(stored.dtype(), Dtype::F16 | Dtype::F32 | Dtype::F64);
    let interpolation = if config.nearest {
        TransformInterpolation::Nearest
    } else {
        TransformInterpolation::Linear
    };
    let warped = warp(
        &stored.widened(),
        &source.frame(level),
        lattice,
        lattice_shape,
        map,
        interpolation,
    )?;
    let image = if config.nearest && integer {
        Voxels::from(warped.mapv(|v| v.max(0.0) as u64))
    } else {
        Voxels::from(warped.mapv(|v| v as f32))
    };
    write_image(path, &image, lattice, source.unit().as_deref())
}

fn frame_json(frame: &Frame) -> Value {
    json!({ "scale": frame.scale, "translation": frame.translation })
}

fn side_json(input: &Input, volume: &Volume, level: usize) -> Value {
    json!({
        "path": input.path.display().to_string(),
        "channel": input.channel,
        "level": level,
        "level0_shape": volume.shape(0),
        "level0_frame": frame_json(&volume.frame(0)),
        "unit": volume.unit(),
    })
}

// ------------------------------------------------------------- points --

/// What the coordinates in a point table mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Units {
    /// Level-0 voxel indices of the image the table belongs to.
    Voxel,
    /// Physical coordinates, in the images' unit.
    Physical,
}

/// Where a point table's transform comes from.
#[derive(Debug, Clone)]
pub enum PointSource {
    /// A directory `register` wrote: the transform and both images' frames.
    Registration(PathBuf),
    /// A bare elastix transform file, applied in `space`, with no frames.
    Transform { file: PathBuf, space: PointSpace },
}

#[derive(Debug, Clone)]
pub struct PointsConfig {
    pub source: PointSource,
    pub input: PathBuf,
    pub output: PathBuf,
    /// The three coordinate column names, `[z, y, x]`.
    pub columns: [String; 3],
    /// Prefix for the three appended output columns.
    pub prefix: String,
    /// Units of the input and of the output, with a registration directory.
    pub units: Units,
    /// Reverse each point before and after a bare transform file, for one
    /// written in `x, y, z` order (by elastix from an `x, y, z` array, say).
    /// Refused with a registration directory, whose files are `z, y, x`.
    pub reverse_axes: bool,
    /// Map moving (atlas) voxels back to fixed (sample) voxels, by inverting
    /// the transform numerically. Registration directories in voxel units.
    pub inverse: bool,
}

/// The point map a registration directory describes: fixed level-0 voxels (or
/// physical coordinates) to moving level-0 voxels (or physical coordinates).
pub fn registration_transform(directory: &Path, units: Units) -> Result<AtlasTransform> {
    let text = fs::read_to_string(directory.join(REGISTRATION_JSON)).map_err(|error| {
        Error::invalid(format!(
            "{}: not a registration directory ({REGISTRATION_JSON}: {error})",
            directory.display()
        ))
    })?;
    let sidecar: Value = serde_json::from_str(&text).map_err(Error::backend)?;
    let name = sidecar
        .get("transform")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invalid(format!("{REGISTRATION_JSON} names no transform file")))?;
    let transform = ElastixTransform::read(directory.join(name))?;
    let (into, out_of) = match units {
        Units::Physical => (Rescale::identity(), Rescale::identity()),
        Units::Voxel => (
            Rescale::to_physical(&sidecar_frame(&sidecar, "fixed")?),
            Rescale::to_index(&sidecar_frame(&sidecar, "moving")?),
        ),
    };
    Ok(AtlasTransform::new(
        into,
        vec![transform],
        out_of,
        RegisterParams::default(),
    ))
}

fn sidecar_frame(sidecar: &Value, side: &str) -> Result<Frame> {
    let triple = |key: &str| -> Result<[f64; 3]> {
        let values = sidecar
            .pointer(&format!("/{side}/level0_frame/{key}"))
            .and_then(Value::as_array)
            .filter(|values| values.len() == 3)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{REGISTRATION_JSON}: {side}.level0_frame.{key} is missing"
                ))
            })?;
        let mut out = [0.0; 3];
        for axis in 0..3 {
            out[axis] = values[axis].as_f64().ok_or_else(|| {
                Error::invalid(format!(
                    "{REGISTRATION_JSON}: {side}.level0_frame.{key} is not numeric"
                ))
            })?;
        }
        Ok(out)
    };
    Frame::new(triple("scale")?, triple("translation")?)
}

/// Transform a CSV point table. Every input column is kept and three columns
/// are appended; returns the number of rows written.
pub fn transform_points(config: &PointsConfig) -> Result<usize> {
    let map: Box<dyn Fn([f64; 3]) -> Option<[f64; 3]>> = match &config.source {
        PointSource::Registration(_) if config.reverse_axes => {
            return Err(Error::invalid(
                "transform-points: --reverse-axes applies to a bare transform file; a \
                 registration directory's files are already in z, y, x order",
            ));
        }
        PointSource::Registration(directory) if config.units == Units::Voxel => {
            // One registration or a chain, forward or back.
            let transform = crate::points::PointTransform::from_registration_dir(directory)?;
            if config.inverse {
                Box::new(move |p| transform.apply_inverse(&[p])[0])
            } else {
                Box::new(move |p| Some(transform.apply_one(p)))
            }
        }
        _ if config.inverse => {
            return Err(Error::invalid(
                "transform-points: --inverse needs a registration directory in voxel units",
            ));
        }
        PointSource::Registration(directory) => {
            if directory.join(CHAIN_JSON).exists() {
                return Err(Error::invalid(
                    "transform-points: a chain maps voxels of its first image to voxels of its \
                     last; physical units are per registration",
                ));
            }
            let transform = registration_transform(directory, config.units)?;
            Box::new(move |p| Some(transform.apply(p)))
        }
        PointSource::Transform { file, space } => {
            let transform = AtlasTransform::new(
                Rescale::identity(),
                vec![ElastixTransform::read(file)?],
                Rescale::identity(),
                RegisterParams::in_space(*space),
            );
            Box::new(move |p| Some(transform.apply(p)))
        }
    };
    let mut reader = csv::Reader::from_path(&config.input).map_err(Error::backend)?;
    let headers = reader.headers().map_err(Error::backend)?.clone();
    let mut index = [0usize; 3];
    for axis in 0..3 {
        index[axis] = headers
            .iter()
            .position(|name| name == config.columns[axis])
            .ok_or_else(|| {
                Error::invalid(format!(
                    "{}: no column `{}` (columns: {})",
                    config.input.display(),
                    config.columns[axis],
                    headers.iter().collect::<Vec<_>>().join(", ")
                ))
            })?;
    }
    let mut writer = csv::Writer::from_path(&config.output).map_err(Error::backend)?;
    let mut out_headers = headers.clone();
    for name in &config.columns {
        out_headers.push_field(&format!("{}{name}", config.prefix));
    }
    writer.write_record(&out_headers).map_err(Error::backend)?;
    let mut rows = 0;
    for (row, record) in reader.records().enumerate() {
        let mut record = record.map_err(Error::backend)?;
        let mut point = [0.0; 3];
        for axis in 0..3 {
            let field = record.get(index[axis]).unwrap_or("");
            point[axis] = field.trim().parse::<f64>().map_err(|error| {
                Error::invalid(format!(
                    "{}: row {}: `{}` = `{field}` is not a number: {error}",
                    config.input.display(),
                    row + 1,
                    config.columns[axis]
                ))
            })?;
        }
        if config.reverse_axes {
            point.reverse();
        }
        match map(point) {
            Some(mut moved) => {
                if config.reverse_axes {
                    moved.reverse();
                }
                for value in moved {
                    record.push_field(&value.to_string());
                }
            }
            None => {
                for _ in 0..3 {
                    record.push_field("NaN");
                }
            }
        }
        writer.write_record(&record).map_err(Error::backend)?;
        rows += 1;
    }
    writer.flush().map_err(Error::backend)?;
    Ok(rows)
}
