// SPDX-License-Identifier: MIT
//
//! Out-of-core intensity pyramids for an **existing** OME-Zarr image.
//!
//! [`plan_pyramid`] reads an image group's multiscale metadata and decides the
//! missing levels; [`build_pyramid`] writes them. Level 0 and every level that
//! already exists are only ever read. Each new level is computed from the level
//! before it, and the group's `multiscales[0].datasets` list is extended last,
//! by a byte splice that leaves every other byte of the (often very large)
//! `zarr.json` untouched.
//!
//! This is the numeric sibling of [`crate::label_pyramid`] (nearest, labels,
//! 3-D, through the planner) and the out-of-core sibling of
//! [`crate::zarr_env::ZarrEnvironment::build_multiscale`] (mean, resident
//! input). It works directly on `zarrs` arrays of any rank, because an OME-Zarr
//! image is `[t][c][z]yx` and the planner's blocks are rank three.
//!
//! # Factor rule (anisotropy-aware)
//!
//! Per level, over the spatial axes whose extent is still > 1, with physical
//! spacing `d` taken from the level's `scale`: an axis is halved iff
//! `d < 1.5 * min(d)`. So `0.1 x 0.1 x 1 µm` voxels halve `y, x` only until
//! they reach `0.8 µm`, after which all three axes halve together; isotropic
//! data halves everything every level. Non-spatial axes (`t`, `c`, anything
//! not `type: space`) are never downsampled. Levels are added until the largest
//! spatial extent is `<= target_size`, or the pyramid has `max_levels` levels.
//!
//! # Values
//!
//! A new voxel is the **mean** (default) or **max** of its window in the
//! previous level, where windows at the far edge are clipped rather than padded
//! (so a level's extent is `ceil(n / f)`). Means are accumulated in `f64` and
//! integers are rounded half away from zero; for `2^k` windows of integer
//! input every intermediate is exact, so the result equals `round(sum/count)`.
//!
//! # Coordinate transformations
//!
//! `scale` is multiplied by the factor. Integer indices locate voxel centres.
//! By default `translation` moves by `(f - 1) / 2 * previous_scale`, locating
//! the centre of each nominal pooling window. This also applies to max pooling:
//! the output represents the window, not the location of its winning sample.
//! Clipped edge windows retain this regular lattice; their available samples'
//! centroid can differ from the nominal centre.
//! [`PyramidOptions::half_voxel_translation`] may be set to `false` for legacy
//! scale-only compatibility. That opt-out does not change the pooling kernel
//! and intentionally loses voxel-centre alignment across levels. It is not the
//! convention for first-sample decimation, which is a different sampling method.
//!
//! # Memory and parallelism
//!
//! Work is grouped by **output write unit** (the shard, or the chunk when
//! unsharded): each unit is assembled in one buffer and written once with
//! `store_chunk`, so no two workers ever write the same shard and no
//! read-modify-write happens. A unit is filled by sub-blocks of the output
//! chunk shape, each read from the previous level as one region. Units are
//! processed in waves; peak memory is about
//!
//! `wave_units * 2 * unit_bytes + workers * chunk_voxels * prod(f) * (elem + 12)`
//!
//! (buffer plus encoded copy per unit, plus each worker's input sub-block, its
//! `f64` copy and the first reduction). `wave_units` is per level: just enough
//! units that a wave holds about `8 * workers` sub-blocks (so every worker is
//! busy even when one shard holds hundreds of chunks), and never more than
//! fits in half of [`PyramidOptions::memory_budget`].

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use serde_json::{json, Value as Json};
use zarrs::array::{Array as ZarrArray, ArraySubset, Element, ElementOwned};
use zarrs::filesystem::FilesystemStore;

use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::zarr_env::Compression;
use crate::zarr_layout::StorageLayout;

/// How a window of the previous level becomes one voxel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Downsample {
    /// The mean, rounded half away from zero for integer types.
    Mean,
    /// The maximum (NaN-ignoring for floats).
    Max,
}

/// A shard override.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardChoice {
    /// Write unsharded arrays.
    None,
    /// Shard shape, spatial axes only or full rank.
    Shape(Vec<usize>),
}

/// What [`plan_pyramid`] and [`build_pyramid`] are asked to do.
#[derive(Clone, Debug)]
pub struct PyramidOptions {
    pub method: Downsample,
    /// Stop once the largest spatial extent is at most this.
    pub target_size: usize,
    /// Maximum total number of levels, existing ones included.
    pub max_levels: Option<usize>,
    /// Chunk (inner chunk when sharded) override: the spatial axes in order,
    /// or the full rank. Non-spatial axes get 1 when only spatial is given.
    pub chunk: Option<Vec<usize>>,
    /// Shard override; `None` keeps level 0's sharding.
    pub shard: Option<ShardChoice>,
    /// Compression override; `None` keeps level 0's.
    pub compression: Option<Compression>,
    /// Worker threads. `0` means the machine's available parallelism.
    pub workers: usize,
    /// Replace target directories that exist but are not listed levels.
    pub overwrite: bool,
    /// Locate nominal pooling-window centres (default). `false` preserves legacy
    /// scale-only placement despite pooling; see the module docs.
    pub half_voxel_translation: bool,
    /// Bytes the build may hold in flight. `0` means
    /// [`crate::budget::default_budget_bytes`].
    pub memory_budget: u64,
    /// Write to this new group instead of in place. Existing levels are
    /// symlinked into it (not copied), new levels are written into it, and its
    /// `zarr.json` is the source's with the new datasets appended.
    pub output: Option<PathBuf>,
}

impl Default for PyramidOptions {
    fn default() -> Self {
        Self {
            method: Downsample::Mean,
            target_size: 256,
            max_levels: None,
            chunk: None,
            shard: None,
            compression: None,
            workers: 0,
            overwrite: false,
            half_voxel_translation: true,
            memory_budget: 0,
            output: None,
        }
    }
}

/// One NGFF axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub name: String,
    /// Only spatial axes are ever downsampled.
    pub spatial: bool,
}

/// A level that exists or will.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedLevel {
    pub path: String,
    pub shape: Vec<usize>,
    /// Factor relative to the previous level (all ones for existing levels).
    pub factor: Vec<usize>,
    pub scale: Vec<f64>,
    pub translation: Option<Vec<f64>>,
    /// The layout it is (to be) written with.
    pub layout: StorageLayout,
}

/// Everything [`build_pyramid`] needs, and everything `--dry-run` prints.
#[derive(Clone, Debug)]
pub struct PyramidPlan {
    /// The image group the levels are read from.
    pub source: PathBuf,
    /// The group written to: `source` in place, or the separate output.
    pub target: PathBuf,
    pub axes: Vec<Axis>,
    pub dtype: Dtype,
    pub existing: Vec<PlannedLevel>,
    pub new_levels: Vec<PlannedLevel>,
    /// Paths among `new_levels` whose directories already exist.
    pub conflicts: Vec<String>,
    pub method: Downsample,
    pub workers: usize,
    /// Units (shards or chunks) per wave, per new level; see the module's
    /// memory section.
    pub wave_units: Vec<usize>,
    fill_value: Json,
    dimension_names: Option<Json>,
    /// `/attributes/ome/multiscales/0` or `/attributes/multiscales/0`.
    multiscale_pointer: String,
}

impl PyramidPlan {
    /// Uncompressed bytes of one new level.
    pub fn level_bytes(&self, level: &PlannedLevel) -> u64 {
        level.shape.iter().product::<usize>() as u64 * self.dtype.size_of() as u64
    }

    /// Estimated peak bytes held in flight while building.
    pub fn estimated_peak_bytes(&self) -> u64 {
        let elem = self.dtype.size_of() as u64;
        self.new_levels
            .iter()
            .zip(&self.wave_units)
            .map(|(level, &wave_units)| {
                let unit = level.layout.write_unit().iter().product::<usize>() as u64 * elem;
                let window: usize = level.factor.iter().product();
                let sub = level.layout.chunk.iter().product::<usize>() as u64
                    * window as u64
                    * (elem + 12);
                wave_units as u64 * 2 * unit + self.workers as u64 * sub
            })
            .max()
            .unwrap_or(0)
    }

    /// The plan as text, one level per line.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        let names: Vec<&str> = self.axes.iter().map(|axis| axis.name.as_str()).collect();
        out.push_str(&format!(
            "image {}\n  target {}\n  axes {:?}, dtype {}, method {:?}\n",
            self.source.display(),
            self.target.display(),
            names,
            self.dtype.numpy_name(),
            self.method
        ));
        let line = |level: &PlannedLevel, tag: &str| {
            format!(
                "  {tag} {:>3}: shape {:?} factor {:?} scale {:?}{} {} ({})\n",
                level.path,
                level.shape,
                level.factor,
                level.scale,
                level
                    .translation
                    .as_ref()
                    .map(|t| format!(" translation {t:?}"))
                    .unwrap_or_default(),
                level.layout.describe(),
                human_bytes(self.level_bytes(level))
            )
        };
        for level in &self.existing {
            out.push_str(&line(level, "existing"));
        }
        for level in &self.new_levels {
            out.push_str(&line(level, "new     "));
        }
        if self.new_levels.is_empty() {
            out.push_str("  nothing to do: the pyramid already reaches the target size\n");
        }
        let total: u64 = self.new_levels.iter().map(|l| self.level_bytes(l)).sum();
        out.push_str(&format!(
            "  new data {} uncompressed; {} workers, units per wave {:?}, est. peak memory {}\n",
            human_bytes(total),
            self.workers,
            self.wave_units,
            human_bytes(self.estimated_peak_bytes())
        ));
        if !self.conflicts.is_empty() {
            out.push_str(&format!(
                "  WARNING: target paths already exist: {:?} (use --overwrite)\n",
                self.conflicts
            ));
        }
        out
    }
}

/// Progress of [`build_pyramid`]: called after every wave.
#[derive(Clone, Debug)]
pub struct PyramidProgress {
    /// Index into `plan.new_levels`.
    pub level: usize,
    pub levels: usize,
    pub path: String,
    pub units_done: usize,
    pub units_total: usize,
}

// ------------------------------------------------------------- the plan --

/// Factor for the next level, or `None` when no axis can shrink.
///
/// `spatial[a]` marks the axes that may be downsampled; see the module docs
/// for the rule.
pub fn next_factor(shape: &[usize], spacing: &[f64], spatial: &[bool]) -> Option<Vec<usize>> {
    let candidates: Vec<usize> = (0..shape.len())
        .filter(|&axis| spatial[axis] && shape[axis] > 1)
        .collect();
    let finest = candidates
        .iter()
        .map(|&axis| positive(spacing[axis]))
        .fold(f64::INFINITY, f64::min);
    if candidates.is_empty() {
        return None;
    }
    let mut factor = vec![1; shape.len()];
    for &axis in &candidates {
        if positive(spacing[axis]) < 1.5 * finest {
            factor[axis] = 2;
        }
    }
    Some(factor)
}

/// The factors of every level after `shape`, until the largest spatial extent
/// is `<= target` or `max_new` levels have been planned.
pub fn plan_factors(
    shape: &[usize],
    spacing: &[f64],
    spatial: &[bool],
    target: usize,
    max_new: usize,
) -> Vec<Vec<usize>> {
    let mut shape = shape.to_vec();
    let mut spacing = spacing.to_vec();
    let mut factors = Vec::new();
    while factors.len() < max_new {
        let largest = (0..shape.len())
            .filter(|&axis| spatial[axis])
            .map(|axis| shape[axis])
            .max()
            .unwrap_or(0);
        if largest <= target.max(1) {
            break;
        }
        let Some(factor) = next_factor(&shape, &spacing, spatial) else {
            break;
        };
        for axis in 0..shape.len() {
            shape[axis] = shape[axis].div_ceil(factor[axis]);
            spacing[axis] = positive(spacing[axis]) * factor[axis] as f64;
        }
        factors.push(factor);
    }
    factors
}

fn positive(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        1.0
    }
}

/// Read `image`'s multiscale metadata and decide the levels to add.
///
/// `image` may be the image group itself or a bioformats2raw-style container
/// whose `OME/zarr.json` names the series.
pub fn plan_pyramid(image: &Path, options: &PyramidOptions) -> Result<PyramidPlan> {
    let (source, metadata) = crate::ome_zarr::locate_image_group(image)?;
    if metadata.get("zarr_format").and_then(Json::as_u64) != Some(3) {
        return Err(Error::invalid(format!(
            "{} is not a Zarr v3 group; NGFF 0.4 on Zarr v2 is not supported",
            source.display()
        )));
    }
    let multiscale_pointer = if metadata.pointer("/attributes/ome/multiscales/0").is_some() {
        "/attributes/ome/multiscales/0"
    } else {
        "/attributes/multiscales/0"
    }
    .to_owned();
    let multiscale = metadata
        .pointer(&multiscale_pointer)
        .ok_or_else(|| Error::invalid("image group has no multiscales entry"))?;
    let axes = parse_axes(multiscale)?;
    let spatial: Vec<bool> = axes.iter().map(|axis| axis.spatial).collect();
    let datasets = multiscale
        .get("datasets")
        .and_then(Json::as_array)
        .filter(|datasets| !datasets.is_empty())
        .ok_or_else(|| Error::invalid("multiscales entry has no datasets"))?;

    let mut existing = Vec::with_capacity(datasets.len());
    let mut level0: Option<(Json, Dtype)> = None;
    for dataset in datasets {
        let path = dataset
            .get("path")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::invalid("multiscale dataset has no path"))?
            .to_owned();
        let array = read_json(&source.join(&path).join("zarr.json"))?;
        let shape = json_usizes(array.get("shape"), &path)?;
        if shape.len() != axes.len() {
            return Err(Error::invalid(format!(
                "level {path} has rank {} but {} axes are declared",
                shape.len(),
                axes.len()
            )));
        }
        let (scale, translation) = parse_transforms(dataset, axes.len(), &path)?;
        let layout = StorageLayout::of_metadata(&array, &path)?;
        if level0.is_none() {
            let dtype = array
                .get("data_type")
                .and_then(Json::as_str)
                .and_then(Dtype::from_numpy_name)
                .filter(|dtype| !matches!(dtype, Dtype::Bool | Dtype::F16))
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "level {path} has data type {}, which an intensity pyramid does not support",
                        array.get("data_type").unwrap_or(&Json::Null)
                    ))
                })?;
            level0 = Some((array.clone(), dtype));
        }
        existing.push(PlannedLevel {
            path,
            factor: vec![1; shape.len()],
            shape,
            scale,
            translation,
            layout,
        });
    }
    let (array0, dtype) = level0.expect("at least one dataset");
    let base_layout = override_layout(&existing[0].layout, &spatial, options)?;

    let last = existing.last().expect("at least one dataset").clone();
    let max_new = options
        .max_levels
        .map(|max| max.saturating_sub(existing.len()))
        .unwrap_or(32);
    let factors = plan_factors(
        &last.shape,
        &last.scale,
        &spatial,
        options.target_size,
        max_new,
    );

    let mut next_index = existing
        .iter()
        .filter_map(|level| level.path.parse::<usize>().ok())
        .max()
        .map_or(existing.len(), |max| max + 1);
    let target = options.output.clone().unwrap_or_else(|| source.clone());
    let mut new_levels = Vec::with_capacity(factors.len());
    let mut conflicts = Vec::new();
    let mut previous = last;
    for factor in factors {
        while existing
            .iter()
            .any(|level| level.path == next_index.to_string())
        {
            next_index += 1;
        }
        let path = next_index.to_string();
        next_index += 1;
        let shape: Vec<usize> = previous
            .shape
            .iter()
            .zip(&factor)
            .map(|(&extent, &f)| extent.div_ceil(f))
            .collect();
        let scale: Vec<f64> = previous
            .scale
            .iter()
            .zip(&factor)
            .map(|(&scale, &f)| scale * f as f64)
            .collect();
        let translation = if options.half_voxel_translation {
            let base = previous
                .translation
                .clone()
                .unwrap_or_else(|| vec![0.0; shape.len()]);
            let shifted: Vec<f64> = (0..shape.len())
                .map(|axis| {
                    crate::pyramid_geometry::pyramid_axis_transform(
                        base[axis],
                        previous.scale[axis],
                        factor[axis],
                        crate::pyramid_geometry::PyramidSampling::WindowCentre,
                    )
                    .1
                })
                .collect();
            (previous.translation.is_some() || shifted.iter().any(|&v| v != 0.0)).then_some(shifted)
        } else {
            previous.translation.clone()
        };
        if target.join(&path).exists() {
            conflicts.push(path.clone());
        }
        let level = PlannedLevel {
            path,
            layout: base_layout.fitted_to(&shape)?,
            shape,
            factor,
            scale,
            translation,
        };
        previous = level.clone();
        new_levels.push(level);
    }

    let workers = if options.workers == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        options.workers
    };
    let budget = if options.memory_budget == 0 {
        crate::budget::default_budget_bytes()
    } else {
        options.memory_budget
    };
    let wave_units = new_levels
        .iter()
        .map(|level| {
            let unit = level.layout.write_unit();
            let unit_bytes = unit.iter().product::<usize>() as u64 * dtype.size_of() as u64;
            let per_unit: usize = unit
                .iter()
                .zip(&level.layout.chunk)
                .map(|(&u, &c)| u.div_ceil(c))
                .product();
            let busy = (8 * workers).div_ceil(per_unit.max(1));
            let fits = ((budget / 2) / (2 * unit_bytes).max(1)).max(1) as usize;
            busy.min(fits).max(1)
        })
        .collect();

    Ok(PyramidPlan {
        source,
        target,
        axes,
        dtype,
        existing,
        new_levels,
        conflicts,
        method: options.method,
        workers,
        wave_units,
        fill_value: array0.get("fill_value").cloned().unwrap_or(json!(0)),
        dimension_names: array0.get("dimension_names").cloned(),
        multiscale_pointer,
    })
}

fn parse_axes(multiscale: &Json) -> Result<Vec<Axis>> {
    multiscale
        .get("axes")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::invalid("multiscales entry has no axes"))?
        .iter()
        .map(|axis| {
            let name = axis
                .as_str()
                .or_else(|| axis.get("name").and_then(Json::as_str))
                .ok_or_else(|| Error::invalid("OME axis has no name"))?
                .to_owned();
            let kind = axis.get("type").and_then(Json::as_str);
            let spatial = match kind {
                Some(kind) => kind == "space",
                None => matches!(name.to_ascii_lowercase().as_str(), "x" | "y" | "z"),
            };
            Ok(Axis { name, spatial })
        })
        .collect()
}

fn parse_transforms(
    dataset: &Json,
    rank: usize,
    path: &str,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let mut scale = vec![1.0; rank];
    let mut translation = None;
    for transform in dataset
        .get("coordinateTransformations")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let kind = transform.get("type").and_then(Json::as_str).unwrap_or("");
        let values = |key: &str| -> Result<Vec<f64>> {
            let values = transform
                .get(key)
                .and_then(Json::as_array)
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "level {path}: {kind} transform without inline values is not supported"
                    ))
                })?
                .iter()
                .map(|value| value.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            if values.len() != rank || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid(format!(
                    "level {path}: {kind} {values:?} does not match rank {rank}"
                )));
            }
            Ok(values)
        };
        match kind {
            "scale" => scale = values("scale")?,
            "translation" => translation = Some(values("translation")?),
            "identity" => {}
            other => {
                return Err(Error::invalid(format!(
                    "level {path}: unsupported coordinate transformation `{other}`"
                )))
            }
        }
    }
    Ok((scale, translation))
}

/// Apply the chunk / shard / compression overrides to level 0's layout.
fn override_layout(
    level0: &StorageLayout,
    spatial: &[bool],
    options: &PyramidOptions,
) -> Result<StorageLayout> {
    let rank = spatial.len();
    let expand = |given: &[usize], what: &str| -> Result<Vec<usize>> {
        let spatial_count = spatial.iter().filter(|&&s| s).count();
        if given.contains(&0) {
            return Err(Error::invalid(format!("{what} {given:?} must be positive")));
        }
        if given.len() == rank {
            return Ok(given.to_vec());
        }
        if given.len() == spatial_count {
            let mut values = given.iter();
            return Ok(spatial
                .iter()
                .map(|&s| {
                    if s {
                        *values.next().expect("counted")
                    } else {
                        1
                    }
                })
                .collect());
        }
        Err(Error::invalid(format!(
            "{what} {given:?} must list the {spatial_count} spatial axes or all {rank} axes"
        )))
    };
    let mut layout = level0.clone();
    if let Some(chunk) = &options.chunk {
        layout.chunk = expand(chunk, "chunk")?;
    }
    match &options.shard {
        None => {}
        Some(ShardChoice::None) => layout.shard = None,
        Some(ShardChoice::Shape(shard)) => {
            layout.shard = Some(expand(shard, "shard")?);
            if level0.shard.is_none() {
                layout.index_checksum = true;
            }
        }
    }
    if let Some(shard) = &mut layout.shard {
        // A shard must hold whole chunks.
        for (shard, &chunk) in shard.iter_mut().zip(&layout.chunk) {
            *shard = (*shard).max(chunk).div_ceil(chunk) * chunk;
        }
    }
    if let Some(compression) = options.compression {
        layout.compression = compression;
    }
    Ok(layout)
}

// ------------------------------------------------------------ the build --

/// Build every level of `plan` and then register them in the group metadata.
///
/// Returns the paths written. Refuses when a target path exists and
/// `options.overwrite` is false; with it, such a directory is removed first
/// (it can never be a listed level — those are never chosen as targets).
pub fn build_pyramid(
    plan: &PyramidPlan,
    options: &PyramidOptions,
    progress: &(dyn Fn(&PyramidProgress) + Sync),
) -> Result<Vec<String>> {
    if plan.new_levels.is_empty() {
        return Ok(Vec::new());
    }
    if !plan.conflicts.is_empty() && !options.overwrite {
        return Err(Error::invalid(format!(
            "{}: level paths {:?} already exist but are not listed levels (an interrupted run?); \
             pass --overwrite to replace them",
            plan.target.display(),
            plan.conflicts
        )));
    }
    if plan.target != plan.source {
        prepare_separate_output(plan, options.overwrite)?;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(plan.workers.max(1))
        .build()
        .map_err(Error::backend)?;
    let store = Arc::new(FilesystemStore::new(&plan.target).map_err(Error::backend)?);

    let mut previous_path = plan.existing.last().expect("level 0").path.clone();
    for (index, level) in plan.new_levels.iter().enumerate() {
        let dir = plan.target.join(&level.path);
        if dir.exists() {
            fs::remove_dir_all(&dir).map_err(Error::backend)?;
        }
        fs::create_dir_all(&dir).map_err(Error::backend)?;
        let metadata = level.layout.array_metadata(
            &level.shape,
            plan.dtype,
            plan.fill_value.clone(),
            plan.dimension_names.clone(),
        )?;
        fs::write(
            dir.join("zarr.json"),
            serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
        )
        .map_err(Error::backend)?;
        let input =
            ZarrArray::open(store.clone(), &format!("/{previous_path}")).map_err(Error::backend)?;
        let output =
            ZarrArray::open(store.clone(), &format!("/{}", level.path)).map_err(Error::backend)?;
        let job = LevelJob {
            input: &input,
            output: &output,
            level,
            levels: plan.new_levels.len(),
            index,
            method: plan.method,
            fill: fill_f64(&plan.fill_value),
            wave_units: plan.wave_units[index],
            progress,
        };
        pool.install(|| dispatch(plan.dtype, &job))?;
        previous_path = level.path.clone();
    }

    append_datasets(plan)?;
    Ok(plan
        .new_levels
        .iter()
        .map(|level| level.path.clone())
        .collect())
}

struct LevelJob<'a> {
    input: &'a ZarrArray<FilesystemStore>,
    output: &'a ZarrArray<FilesystemStore>,
    level: &'a PlannedLevel,
    levels: usize,
    index: usize,
    method: Downsample,
    fill: f64,
    wave_units: usize,
    progress: &'a (dyn Fn(&PyramidProgress) + Sync),
}

fn dispatch(dtype: Dtype, job: &LevelJob<'_>) -> Result<()> {
    match dtype {
        Dtype::U8 => build_level::<u8>(job),
        Dtype::U16 => build_level::<u16>(job),
        Dtype::U32 => build_level::<u32>(job),
        Dtype::U64 => build_level::<u64>(job),
        Dtype::I8 => build_level::<i8>(job),
        Dtype::I16 => build_level::<i16>(job),
        Dtype::I32 => build_level::<i32>(job),
        Dtype::I64 => build_level::<i64>(job),
        Dtype::F32 => build_level::<f32>(job),
        Dtype::F64 => build_level::<f64>(job),
        Dtype::Bool | Dtype::F16 => Err(Error::invalid(format!(
            "{} intensity pyramids are not supported",
            dtype.numpy_name()
        ))),
    }
}

/// An element an intensity pyramid can hold.
trait Sample: Element + ElementOwned + Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    /// Rounded (integers) and saturated into range.
    fn from_f64(value: f64) -> Self;
}

macro_rules! integer_sample {
    ($($t:ty),*) => {$(
        impl Sample for $t {
            fn to_f64(self) -> f64 { self as f64 }
            fn from_f64(value: f64) -> Self { value.round() as $t }
        }
    )*};
}
integer_sample!(u8, u16, u32, u64, i8, i16, i32, i64);

impl Sample for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl Sample for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}

fn build_level<T: Sample>(job: &LevelJob<'_>) -> Result<()> {
    let level = job.level;
    let rank = level.shape.len();
    let unit = level.layout.write_unit().to_vec();
    let grid: Vec<usize> = level
        .shape
        .iter()
        .zip(&unit)
        .map(|(&extent, &unit)| extent.div_ceil(unit))
        .collect();
    let units: Vec<Vec<usize>> = boxes(&vec![0; rank], &grid, &vec![1; rank])
        .into_iter()
        .map(|(start, _)| start)
        .collect();
    let in_shape: Vec<usize> = job.input.shape().iter().map(|&v| v as usize).collect();
    let unit_len: usize = unit.iter().product();
    let fill = T::from_f64(job.fill);
    let done = AtomicUsize::new(0);

    for wave in units.chunks(job.wave_units.max(1)) {
        // One buffer per unit, the unbounded unit shape, filled with the fill
        // value so the part past the array's edge is what Zarr expects there.
        let buffers: Vec<Mutex<Vec<T>>> = wave
            .iter()
            .map(|_| Mutex::new(vec![fill; unit_len]))
            .collect();
        let tasks: Vec<(usize, Vec<usize>, Vec<usize>)> = wave
            .iter()
            .enumerate()
            .flat_map(|(slot, index)| {
                let start: Vec<usize> = index.iter().zip(&unit).map(|(&i, &u)| i * u).collect();
                let end: Vec<usize> = start
                    .iter()
                    .zip(&unit)
                    .zip(&level.shape)
                    .map(|((&s, &u), &extent)| (s + u).min(extent))
                    .collect();
                boxes(&start, &end, &level.layout.chunk)
                    .into_iter()
                    .map(move |(sub_start, sub_shape)| (slot, sub_start, sub_shape))
            })
            .collect();
        tasks
            .par_iter()
            .try_for_each(|(slot, sub_start, sub_shape)| -> Result<()> {
                let values = downsample_block::<T>(
                    job.input,
                    &in_shape,
                    sub_start,
                    sub_shape,
                    &level.factor,
                    job.method,
                )?;
                let offset: Vec<usize> = sub_start
                    .iter()
                    .zip(&wave[*slot])
                    .zip(&unit)
                    .map(|((&s, &i), &u)| s - i * u)
                    .collect();
                let mut buffer = buffers[*slot].lock().unwrap_or_else(|p| p.into_inner());
                copy_block(&mut buffer, &unit, &offset, &values, sub_shape);
                Ok(())
            })?;
        buffers.into_par_iter().zip(wave.par_iter()).try_for_each(
            |(buffer, index)| -> Result<()> {
                let buffer = buffer.into_inner().unwrap_or_else(|p| p.into_inner());
                let indices: Vec<u64> = index.iter().map(|&i| i as u64).collect();
                job.output
                    .store_chunk(&indices, buffer.as_slice())
                    .map_err(|error| {
                        Error::backend(format!(
                            "writing level {} unit {index:?}: {error}",
                            level.path
                        ))
                    })
            },
        )?;
        let units_done = done.fetch_add(wave.len(), Ordering::Relaxed) + wave.len();
        (job.progress)(&PyramidProgress {
            level: job.index,
            levels: job.levels,
            path: level.path.clone(),
            units_done,
            units_total: units.len(),
        });
    }
    Ok(())
}

/// Read the input window of one output box and reduce it.
fn downsample_block<T: Sample>(
    input: &ZarrArray<FilesystemStore>,
    in_shape: &[usize],
    out_start: &[usize],
    out_shape: &[usize],
    factor: &[usize],
    method: Downsample,
) -> Result<Vec<T>> {
    let rank = out_start.len();
    let mut start = Vec::with_capacity(rank);
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lo = out_start[axis] * factor[axis];
        let hi = ((out_start[axis] + out_shape[axis]) * factor[axis]).min(in_shape[axis]);
        start.push(lo as u64);
        shape.push((hi - lo) as u64);
    }
    let subset = ArraySubset::new_with_start_shape(start, shape.clone()).map_err(Error::backend)?;
    let raw: Vec<T> = input
        .retrieve_array_subset(&subset)
        .map_err(Error::backend)?;
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    if factor.iter().all(|&f| f == 1) {
        return Ok(raw);
    }
    let (values, reduced) = reduce(&raw, &shape, factor, method);
    debug_assert_eq!(reduced, out_shape);
    Ok(values.into_iter().map(T::from_f64).collect())
}

/// Separable reduction of a C-order block by `factor`, clipping windows at the
/// far edge. Returns the values in `f64` and the reduced shape.
///
/// A mean over a product window is the iterated mean along each axis, so this
/// reduces one axis at a time; the last axis first, where rows are contiguous.
fn reduce<T: Sample>(
    data: &[T],
    shape: &[usize],
    factor: &[usize],
    method: Downsample,
) -> (Vec<f64>, Vec<usize>) {
    let mut values: Vec<f64> = data.iter().map(|&v| v.to_f64()).collect();
    let mut shape = shape.to_vec();
    for axis in (0..shape.len()).rev() {
        let f = factor[axis];
        if f <= 1 {
            continue;
        }
        let outer: usize = shape[..axis].iter().product();
        let n = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let m = n.div_ceil(f);
        let mut out = vec![0.0; outer * m * inner];
        for o in 0..outer {
            for j in 0..m {
                let lo = j * f;
                let hi = (lo + f).min(n);
                let dest = &mut out[(o * m + j) * inner..(o * m + j + 1) * inner];
                let first = (o * n + lo) * inner;
                dest.copy_from_slice(&values[first..first + inner]);
                for k in lo + 1..hi {
                    let row = &values[(o * n + k) * inner..(o * n + k + 1) * inner];
                    match method {
                        Downsample::Mean => {
                            for (d, &v) in dest.iter_mut().zip(row) {
                                *d += v;
                            }
                        }
                        Downsample::Max => {
                            for (d, &v) in dest.iter_mut().zip(row) {
                                *d = d.max(v);
                            }
                        }
                    }
                }
                if method == Downsample::Mean && hi - lo > 1 {
                    let scale = 1.0 / (hi - lo) as f64;
                    for d in dest.iter_mut() {
                        *d *= scale;
                    }
                }
            }
        }
        values = out;
        shape[axis] = m;
    }
    (values, shape)
}

/// Copy a C-order `src` block of `src_shape` into `dst` (C-order, `dst_shape`)
/// at `offset`.
fn copy_block<T: Copy>(
    dst: &mut [T],
    dst_shape: &[usize],
    offset: &[usize],
    src: &[T],
    src_shape: &[usize],
) {
    let rank = dst_shape.len();
    if rank == 0 || src.is_empty() {
        return;
    }
    let row = src_shape[rank - 1];
    let rows = src.len() / row;
    let mut index = vec![0usize; rank - 1];
    for r in 0..rows {
        let mut at = 0;
        for axis in 0..rank - 1 {
            at = at * dst_shape[axis] + offset[axis] + index[axis];
        }
        at = at * dst_shape[rank - 1] + offset[rank - 1];
        dst[at..at + row].copy_from_slice(&src[r * row..(r + 1) * row]);
        for axis in (0..rank - 1).rev() {
            index[axis] += 1;
            if index[axis] < src_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// Every box of `step` tiling `[start, end)`, clipped at `end`, in C order,
/// as `(start, shape)`.
fn boxes(start: &[usize], end: &[usize], step: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let rank = start.len();
    let mut out = Vec::new();
    if (0..rank).any(|axis| start[axis] >= end[axis]) {
        return out;
    }
    let mut at = start.to_vec();
    loop {
        let shape = (0..rank)
            .map(|axis| (at[axis] + step[axis].max(1)).min(end[axis]) - at[axis])
            .collect();
        out.push((at.clone(), shape));
        let mut axis = rank;
        loop {
            if axis == 0 {
                return out;
            }
            axis -= 1;
            at[axis] += step[axis].max(1);
            if at[axis] < end[axis] {
                break;
            }
            at[axis] = start[axis];
        }
    }
}

fn fill_f64(fill: &Json) -> f64 {
    match fill {
        Json::Number(number) => number.as_f64().unwrap_or(0.0),
        Json::String(text) => match text.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            _ => 0.0,
        },
        Json::Bool(true) => 1.0,
        _ => 0.0,
    }
}

// --------------------------------------------------------- the metadata --

static METADATA_TEMP: AtomicU64 = AtomicU64::new(0);

/// Append the new levels to `datasets` in the target's `zarr.json`.
///
/// A textual splice before the array's closing bracket: every other byte of the
/// document, existing dataset entries included, is left as it was. The result
/// is parsed back and checked to equal the original with only the new entries
/// added, then installed by rename.
fn append_datasets(plan: &PyramidPlan) -> Result<()> {
    let path = plan.target.join("zarr.json");
    let text = fs::read_to_string(&path).map_err(Error::backend)?;
    let original: Json = serde_json::from_str(&text).map_err(Error::backend)?;
    let entries: Vec<Json> = plan
        .new_levels
        .iter()
        .map(|level| {
            let mut transforms = vec![json!({"type": "scale", "scale": level.scale})];
            if let Some(translation) = &level.translation {
                transforms.push(json!({"type": "translation", "translation": translation}));
            }
            json!({"path": level.path, "coordinateTransformations": transforms})
        })
        .collect();

    let mut segments: Vec<&str> = plan
        .multiscale_pointer
        .trim_start_matches('/')
        .split('/')
        .collect();
    segments.push("datasets");
    let spliced = splice_append(&text, &segments, &entries)?;

    let mut expected = original;
    expected
        .pointer_mut(&format!("{}/datasets", plan.multiscale_pointer))
        .and_then(Json::as_array_mut)
        .ok_or_else(|| Error::invalid("datasets vanished"))?
        .extend(entries);
    let reparsed: Json = serde_json::from_str(&spliced).map_err(Error::backend)?;
    if reparsed != expected {
        return Err(Error::backend(
            "internal error: the spliced zarr.json does not round-trip; nothing was changed",
        ));
    }

    let serial = METADATA_TEMP.fetch_add(1, Ordering::Relaxed);
    let temporary = plan.target.join(format!(
        ".zarr.json.blockflow-pyramid-{}-{serial}.tmp",
        std::process::id()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(Error::backend)?;
    file.write_all(spliced.as_bytes()).map_err(Error::backend)?;
    file.sync_all().map_err(Error::backend)?;
    drop(file);
    fs::rename(&temporary, &path).map_err(Error::backend)
}

/// Insert `entries` at the end of the JSON array found by following `segments`
/// (object keys or array indices) from the document root.
fn splice_append(text: &str, segments: &[&str], entries: &[Json]) -> Result<String> {
    let bytes = text.as_bytes();
    let mut at = skip_ws(bytes, 0);
    for segment in segments {
        at = match bytes.get(at) {
            Some(b'{') => object_member(bytes, at, segment)?,
            Some(b'[') => {
                let index: usize = segment
                    .parse()
                    .map_err(|_| Error::invalid(format!("expected an index, got {segment}")))?;
                array_elements(bytes, at)?
                    .get(index)
                    .map(|&(start, _)| start)
                    .ok_or_else(|| Error::invalid(format!("no element {index}")))?
            }
            _ => return Err(Error::invalid(format!("cannot descend into {segment}"))),
        };
    }
    if bytes.get(at) != Some(&b'[') {
        return Err(Error::invalid("datasets is not an array"));
    }
    let elements = array_elements(bytes, at)?;
    let close = value_end(bytes, at)? - 1;
    let Some(&(last_start, last_end)) = elements.last() else {
        return Err(Error::invalid("datasets is empty"));
    };
    let compact = !text[at..close].contains('\n');
    let line_start = text[..last_start].rfind('\n').map_or(0, |n| n + 1);
    let indent: String = text[line_start..last_start]
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();
    let mut insert = String::new();
    for entry in entries {
        if compact {
            insert.push_str(", ");
            insert.push_str(&serde_json::to_string(entry).map_err(Error::backend)?);
        } else {
            insert.push_str(",\n");
            insert.push_str(&indent);
            let pretty = serde_json::to_string_pretty(entry).map_err(Error::backend)?;
            insert.push_str(&pretty.replace('\n', &format!("\n{indent}")));
        }
    }
    let mut out = String::with_capacity(text.len() + insert.len());
    out.push_str(&text[..last_end]);
    out.push_str(&insert);
    out.push_str(&text[last_end..]);
    Ok(out)
}

fn skip_ws(bytes: &[u8], mut at: usize) -> usize {
    while at < bytes.len() && bytes[at].is_ascii_whitespace() {
        at += 1;
    }
    at
}

/// End (exclusive) of the JSON value starting at `at`.
fn value_end(bytes: &[u8], at: usize) -> Result<usize> {
    let bad = || Error::invalid("malformed JSON in zarr.json");
    match bytes.get(at).ok_or_else(bad)? {
        b'"' => {
            let mut i = at + 1;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => i += 2,
                    b'"' => return Ok(i + 1),
                    _ => i += 1,
                }
            }
            Err(bad())
        }
        b'{' | b'[' => {
            let mut depth = 0usize;
            let mut i = at;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => {
                        i = value_end(bytes, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            Err(bad())
        }
        _ => {
            let mut i = at;
            while i < bytes.len()
                && !matches!(bytes[i], b',' | b'}' | b']')
                && !bytes[i].is_ascii_whitespace()
            {
                i += 1;
            }
            Ok(i)
        }
    }
}

/// Start of the value of member `key` in the object at `at`.
fn object_member(bytes: &[u8], at: usize, key: &str) -> Result<usize> {
    let mut i = skip_ws(bytes, at + 1);
    while bytes.get(i) == Some(&b'"') {
        let key_end = value_end(bytes, i)?;
        let name: String = serde_json::from_slice(&bytes[i..key_end]).map_err(Error::backend)?;
        i = skip_ws(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            return Err(Error::invalid("malformed JSON object"));
        }
        let value = skip_ws(bytes, i + 1);
        if name == key {
            return Ok(value);
        }
        i = skip_ws(bytes, value_end(bytes, value)?);
        if bytes.get(i) == Some(&b',') {
            i = skip_ws(bytes, i + 1);
        }
    }
    Err(Error::invalid(format!("zarr.json has no member {key}")))
}

/// `(start, end)` of every element of the array at `at`.
fn array_elements(bytes: &[u8], at: usize) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    let mut i = skip_ws(bytes, at + 1);
    while bytes.get(i).is_some_and(|&b| b != b']') {
        let end = value_end(bytes, i)?;
        out.push((i, end));
        i = skip_ws(bytes, end);
        if bytes.get(i) == Some(&b',') {
            i = skip_ws(bytes, i + 1);
        }
    }
    Ok(out)
}

/// Set up a separate output group: symlinks to the existing levels and a copy
/// of the source `zarr.json` (to be extended by [`append_datasets`]).
fn prepare_separate_output(plan: &PyramidPlan, overwrite: bool) -> Result<()> {
    let out = &plan.target;
    if out.join("zarr.json").exists() && !overwrite {
        return Err(Error::invalid(format!(
            "{} already holds a group; pass --overwrite to replace it",
            out.display()
        )));
    }
    fs::create_dir_all(out).map_err(Error::backend)?;
    for level in &plan.existing {
        let link = out.join(&level.path);
        if link.symlink_metadata().is_ok() {
            if link.is_symlink() || link.is_file() {
                fs::remove_file(&link).map_err(Error::backend)?;
            } else {
                fs::remove_dir_all(&link).map_err(Error::backend)?;
            }
        }
        let source = plan
            .source
            .join(&level.path)
            .canonicalize()
            .map_err(Error::backend)?;
        symlink_dir(&source, &link)?;
    }
    fs::copy(plan.source.join("zarr.json"), out.join("zarr.json")).map_err(Error::backend)?;
    Ok(())
}

#[cfg(unix)]
fn symlink_dir(source: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(source, link).map_err(Error::backend)
}

#[cfg(not(unix))]
fn symlink_dir(_source: &Path, _link: &Path) -> Result<()> {
    Err(Error::invalid(
        "a separate output group needs symlinks; build in place instead",
    ))
}

fn read_json(path: &Path) -> Result<Json> {
    serde_json::from_slice(
        &fs::read(path)
            .map_err(|error| Error::backend(format!("reading {}: {error}", path.display())))?,
    )
    .map_err(|error| Error::backend(format!("reading {}: {error}", path.display())))
}

fn json_usizes(value: Option<&Json>, path: &str) -> Result<Vec<usize>> {
    value
        .and_then(Json::as_array)
        .ok_or_else(|| Error::invalid(format!("level {path} has no shape")))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::invalid(format!("level {path} has an invalid shape")))
        })
        .collect()
}

/// `1.5 GiB`-style.
pub fn human_bytes(bytes: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < units.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", units[unit])
    }
}

/// Parse a CLI compression spelling: `none`, `gzip[N]`, `zstd[N]`, or
/// `blosc-<cname>[N][-noshuffle|-shuffle|-bitshuffle]` (e.g. `blosc-zstd5`).
pub fn parse_compression(text: &str) -> Result<Compression> {
    use crate::zarr_env::{BloscCname, BloscShuffle};
    let bad = || Error::invalid(format!("unknown compression `{text}`"));
    // A known name, then an optional level. By name rather than by "trailing
    // digits", because `lz4` ends in one.
    let split_level = |s: &str, names: &[&str]| -> Option<(String, Option<i64>)> {
        let name = names.iter().find(|name| s.starts_with(**name))?;
        let rest = &s[name.len()..];
        let level = if rest.is_empty() {
            None
        } else {
            Some(rest.parse().ok()?)
        };
        Some(((*name).to_owned(), level))
    };
    if text == "none" || text == "bytes" {
        return Ok(Compression::None);
    }
    if let Some(rest) = text.strip_prefix("blosc-") {
        let mut parts = rest.splitn(2, '-');
        let (name, level) = split_level(
            parts.next().unwrap_or(""),
            &["blosclz", "lz4hc", "lz4", "snappy", "zlib", "zstd"],
        )
        .ok_or_else(bad)?;
        let cname = BloscCname::parse(&name).ok_or_else(bad)?;
        let shuffle = match parts.next() {
            None => BloscShuffle::Byte,
            Some(s) => BloscShuffle::parse(&Json::String(s.to_owned())).ok_or_else(bad)?,
        };
        return Ok(Compression::Blosc {
            cname,
            clevel: level.unwrap_or(5).clamp(0, 9) as u8,
            shuffle,
        });
    }
    let (name, level) = split_level(text, &["gzip", "zstd"]).ok_or_else(bad)?;
    match name.as_str() {
        "gzip" => Ok(Compression::Gzip(level.unwrap_or(1).clamp(0, 9) as u32)),
        "zstd" => Ok(Compression::Zstd(level.unwrap_or(3) as i32, false)),
        _ => Err(bad()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_volumes_halve_every_axis_until_the_target() {
        let spatial = [true, true, true];
        let factors = plan_factors(&[1000, 1000, 1000], &[1.0, 1.0, 1.0], &spatial, 256, 32);
        assert_eq!(factors, vec![vec![2, 2, 2]; 2]);
        // 1000 -> 500 -> 250.
    }

    #[test]
    fn anisotropic_z_waits_until_the_fine_axes_catch_up() {
        let spatial = [true, true, true];
        let factors = plan_factors(&[200, 4000, 4000], &[1.0, 0.1, 0.1], &spatial, 256, 32);
        assert_eq!(
            factors,
            vec![
                vec![1, 2, 2], // 0.2
                vec![1, 2, 2], // 0.4
                vec![1, 2, 2], // 0.8 — now within 1.5x of z
                vec![2, 2, 2], // y, x 500 -> 250; z 200 -> 100
            ]
        );
        // Mildly anisotropic (6, 5, 5) is "roughly isotropic": all axes.
        assert_eq!(
            next_factor(&[1030, 2387, 1422], &[6.0, 5.0, 5.0], &spatial),
            Some(vec![2, 2, 2])
        );
    }

    #[test]
    fn two_d_and_singleton_z_only_shrink_the_plane() {
        let factors = plan_factors(&[1000, 600], &[0.5, 0.5], &[true, true], 256, 32);
        assert_eq!(factors, vec![vec![2, 2]; 2]);
        let factors = plan_factors(&[1, 1000, 1000], &[1.0, 0.1, 0.1], &[true; 3], 256, 32);
        assert_eq!(factors, vec![vec![1, 2, 2]; 2]);
    }

    #[test]
    fn time_and_channel_axes_are_never_downsampled() {
        let spatial = [false, false, true, true, true];
        let factors = plan_factors(
            &[5, 3, 64, 1024, 1024],
            &[1.0, 1.0, 2.0, 0.5, 0.5],
            &spatial,
            256,
            32,
        );
        // z (2.0) never comes within 1.5x of y, x (0.5 -> 1.0) before the
        // plane reaches the target.
        assert_eq!(factors, vec![vec![1, 1, 1, 2, 2]; 2]);
        // And the level cap holds.
        assert_eq!(
            plan_factors(&[5, 3, 64, 1024, 1024], &[1.0; 5], &spatial, 1, 1).len(),
            1
        );
    }

    #[test]
    fn reduction_clips_the_far_edge_and_rounds_half_away_from_zero() {
        // 1 x 3: windows [0,1] and [2].
        let (values, shape) = reduce::<u16>(&[1, 2, 7], &[1, 3], &[1, 2], Downsample::Mean);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(values, vec![1.5, 7.0]);
        assert_eq!(u16::from_f64(1.5), 2);
        let (values, _) = reduce::<u16>(&[1, 2, 7, 3], &[2, 2], &[2, 2], Downsample::Max);
        assert_eq!(values, vec![7.0]);
    }

    #[test]
    fn copy_block_and_boxes_agree_on_c_order() {
        let mut dst = vec![0u8; 4 * 5];
        copy_block(&mut dst, &[4, 5], &[1, 2], &[1, 2, 3, 4], &[2, 2]);
        assert_eq!(&dst[7..9], &[1, 2]);
        assert_eq!(&dst[12..14], &[3, 4]);
        let tiles = boxes(&[0, 0], &[5, 3], &[2, 2]);
        assert_eq!(tiles.len(), 6);
        assert_eq!(tiles[5], (vec![4, 2], vec![1, 1]));
    }

    #[test]
    fn the_splice_appends_without_touching_other_bytes() {
        let text = "{\n  \"a\": \"x]y\",\n  \"m\": [\n    {\n      \"datasets\": [\n        {\"path\": \"0\"}\n      ]\n    }\n  ],\n  \"z\": 1\n}";
        let out = splice_append(text, &["m", "0", "datasets"], &[json!({"path": "1"})]).unwrap();
        let parsed: Json = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["m"][0]["datasets"][1]["path"], "1");
        assert!(out.starts_with("{\n  \"a\": \"x]y\",\n  \"m\": [\n    {\n      \"datasets\": [\n        {\"path\": \"0\"},\n        {"));
        assert!(out.ends_with("\n      ]\n    }\n  ],\n  \"z\": 1\n}"));
        let compact = "{\"m\":[{\"datasets\":[{\"path\":\"0\"}]}]}";
        let out = splice_append(compact, &["m", "0", "datasets"], &[json!({"path": "1"})]).unwrap();
        assert_eq!(
            out,
            "{\"m\":[{\"datasets\":[{\"path\":\"0\"}, {\"path\":\"1\"}]}]}"
        );
    }

    #[test]
    fn compression_spellings_parse() {
        use crate::zarr_env::{BloscCname, BloscShuffle};
        assert_eq!(parse_compression("none").unwrap(), Compression::None);
        assert_eq!(parse_compression("gzip6").unwrap(), Compression::Gzip(6));
        assert_eq!(
            parse_compression("zstd").unwrap(),
            Compression::Zstd(3, false)
        );
        assert_eq!(
            parse_compression("blosc-lz4-bitshuffle").unwrap(),
            Compression::Blosc {
                cname: BloscCname::Lz4,
                clevel: 5,
                shuffle: BloscShuffle::Bit
            }
        );
        assert_eq!(
            parse_compression("blosc-zstd3").unwrap(),
            Compression::Blosc {
                cname: BloscCname::Zstd,
                clevel: 3,
                shuffle: BloscShuffle::Byte
            }
        );
    }
}
