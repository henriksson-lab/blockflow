//! Out-of-core nearest-neighbour pyramids for stored label images.
//!
//! Label IDs cannot be averaged. Each level is therefore sampled with nearest
//! interpolation from the preceding level and written through the normal
//! planner, executor, and Zarr environment. This is separate from
//! [`crate::zarr_env::ZarrEnvironment::build_multiscale`], whose numeric levels
//! use averaging and whose level-0 input is resident.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::decomposition::{Constraints, Decomposition};
use crate::fragment::PhaseWork;
use crate::geometry::BlockGrid;
use crate::op::Chain;
use crate::ops::{resample_phase, Interpolation, Resample, ResampleOp};
use crate::strategy::{execute, predicted_phase_prices, Hints, Workflow};
use crate::zarr_env::ZarrEnvironment;
use crate::{AttachedImage, Dtype, Error, Result};
use serde_json::{json, Value as Json};

static METADATA_TEMP: AtomicU64 = AtomicU64::new(0);

/// Build levels `1..` beside an existing `label_root/0` Zarr label array.
///
/// `level_shapes` includes level 0. A completed destination is retained, so an
/// interrupted pyramid build can resume at the first missing level. Every new
/// level reads its immediate predecessor; no downsample rereads level 0.
pub fn build_nearest_label_pyramid(
    label_root: &Path,
    work: &Path,
    level_shapes: &[[usize; 3]],
    block_candidates: &[usize],
    workers: usize,
) -> Result<()> {
    if level_shapes.is_empty() {
        return Err(Error::invalid("a label pyramid needs level 0"));
    }
    if block_candidates.is_empty() || block_candidates.contains(&0) {
        return Err(Error::invalid(
            "label-pyramid block candidates must be positive",
        ));
    }

    let concurrency = workers.max(1);
    for (index, &output_shape) in level_shapes.iter().enumerate().skip(1) {
        let destination = label_root.join(index.to_string());
        if destination.is_dir() {
            continue;
        }

        let input = AttachedImage::at(label_root.join((index - 1).to_string()));
        let (_, input_shape) = input.metadata()?;
        let resample = Resample::to_extent(input_shape, output_shape, Interpolation::Nearest)?;
        let chain = Chain::op(ResampleOp::new("label-pyramid-nearest", resample));
        let workflow = Workflow::new(chain, input_shape, Dtype::U64);
        let constraints = Constraints {
            expected_concurrency: concurrency,
            block_candidates: block_candidates.to_vec(),
            split_axes: if output_shape[0] == 1 {
                vec![1, 2]
            } else {
                vec![0, 1, 2]
            },
            ..Constraints::default()
        };
        let decomposition = planned_resample(
            &workflow,
            &resample,
            input_shape,
            output_shape,
            &constraints,
        )?;
        let level_work = work.join(format!("pyramid-{index}"));
        let env = ZarrEnvironment::attach(&level_work, &[input])?;
        execute(
            "label pyramid",
            &workflow,
            &decomposition,
            &Hints {
                concurrency,
                ..Hints::default()
            },
            &env,
        )?;
        move_path(&level_work.join("level1"), &destination)?;
    }
    Ok(())
}

/// Rebuild the OME-Zarr label-container registry from published label images.
///
/// A label image is an immediate child group with both `image-label` and
/// `ome.multiscales` metadata. Hidden staging directories and unrelated groups
/// are ignored. Existing unrelated container attributes are retained. The new
/// metadata is installed by rename only after the complete file has been
/// written and synced.
pub fn refresh_label_registry(labels_root: &Path) -> Result<()> {
    fs::create_dir_all(labels_root).map_err(Error::backend)?;
    let mut labels = Vec::new();
    let mut version: Option<String> = None;
    for entry in fs::read_dir(labels_root).map_err(Error::backend)? {
        let entry = entry.map_err(Error::backend)?;
        let file_type = entry.file_type().map_err(Error::backend)?;
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().into_string().map_err(|_| {
            Error::invalid(format!(
                "label path below {} is not UTF-8",
                labels_root.display()
            ))
        })?;
        if name.starts_with('.') {
            continue;
        }
        let metadata_path = entry.path().join("zarr.json");
        let Ok(bytes) = fs::read(&metadata_path) else {
            continue;
        };
        let metadata: Json = serde_json::from_slice(&bytes).map_err(|error| {
            Error::backend(format!("reading {}: {error}", metadata_path.display()))
        })?;
        if metadata.pointer("/attributes/image-label").is_none()
            || metadata.pointer("/attributes/ome/multiscales").is_none()
        {
            continue;
        }
        let child_version = metadata
            .pointer("/attributes/ome/version")
            .and_then(Json::as_str)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "published label {} has no OME version",
                    entry.path().display()
                ))
            })?;
        if let Some(expected) = version.as_deref() {
            if child_version != expected {
                return Err(Error::invalid(format!(
                    "published labels below {} mix OME versions {expected} and {child_version}",
                    labels_root.display()
                )));
            }
        } else {
            version = Some(child_version.to_owned());
        }
        labels.push(name);
    }
    labels.sort_unstable();
    labels.dedup();
    let version = version.ok_or_else(|| {
        Error::invalid(format!(
            "{} contains no published OME-Zarr label images",
            labels_root.display()
        ))
    })?;

    let path = labels_root.join("zarr.json");
    let mut metadata = if path.is_file() {
        serde_json::from_slice::<Json>(&fs::read(&path).map_err(Error::backend)?)
            .map_err(|error| Error::backend(format!("reading {}: {error}", path.display())))?
    } else {
        json!({"zarr_format": 3, "node_type": "group", "attributes": {}})
    };
    if metadata.get("zarr_format").and_then(Json::as_u64) != Some(3)
        || metadata.get("node_type").and_then(Json::as_str) != Some("group")
    {
        return Err(Error::invalid(format!(
            "{} is not a Zarr v3 group",
            path.display()
        )));
    }
    let attributes = metadata
        .get_mut("attributes")
        .and_then(Json::as_object_mut)
        .ok_or_else(|| {
            Error::invalid(format!("{} attributes are not an object", path.display()))
        })?;
    let ome = attributes
        .entry("ome")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            Error::invalid(format!("{} OME metadata is not an object", path.display()))
        })?;
    if let Some(existing) = ome.get("version").and_then(Json::as_str) {
        if existing != version {
            return Err(Error::invalid(format!(
                "{} declares OME version {existing}, but its labels use {version}",
                path.display()
            )));
        }
    }
    ome.insert("version".to_owned(), Json::String(version));
    ome.insert(
        "labels".to_owned(),
        Json::Array(labels.into_iter().map(Json::String).collect()),
    );

    let serial = METADATA_TEMP.fetch_add(1, Ordering::Relaxed);
    let temporary = labels_root.join(format!(
        ".zarr.json.blockflow-{}-{serial}.tmp",
        std::process::id()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(Error::backend)?;
    let bytes = serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?;
    file.write_all(&bytes).map_err(Error::backend)?;
    file.sync_all().map_err(Error::backend)?;
    drop(file);
    fs::rename(&temporary, &path).map_err(|error| {
        Error::backend(format!(
            "installing label registry {}: {error}",
            path.display()
        ))
    })
}

fn planned_resample(
    workflow: &Workflow,
    resample: &Resample,
    input: [usize; 3],
    output: [usize; 3],
    constraints: &Constraints,
) -> Result<Decomposition> {
    let mut best: Option<(f64, usize, Decomposition)> = None;
    for &edge in &constraints.block_candidates {
        let grid = BlockGrid::along(output, &constraints.split_axes, edge)?;
        let phase = resample_phase(
            vec![0],
            vec!["label-pyramid-nearest".to_owned()],
            resample,
            input,
            grid,
        )?;
        let decomposition = Decomposition {
            volume: input,
            dtype: Dtype::U64,
            phases: vec![phase],
            chain_reach: workflow.chain.reach3(&input),
        };
        decomposition.check()?;
        let prices = predicted_phase_prices(
            &workflow.chain,
            &decomposition,
            &[PhaseWork::Pixels],
            &constraints.model,
            constraints.expected_concurrency,
        )?;
        let (cost, makespan) = &prices[0];
        if !constraints.affords_working_set(cost) {
            continue;
        }
        if best.as_ref().is_none_or(|(old, old_edge, _)| {
            (*makespan, std::cmp::Reverse(edge)) < (*old, std::cmp::Reverse(*old_edge))
        }) {
            best = Some((*makespan, edge, decomposition));
        }
    }
    best.map(|(_, _, decomposition)| decomposition)
        .ok_or_else(|| Error::invalid("no label-pyramid block candidate fits the constraints"))
}

fn move_path(from: &Path, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).map_err(Error::backend)?;
    }
    fs::rename(from, to).map_err(|error| {
        Error::backend(format!(
            "move {} to {}: {error}",
            from.display(),
            to.display()
        ))
    })
}
