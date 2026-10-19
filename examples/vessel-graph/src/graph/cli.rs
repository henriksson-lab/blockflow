// SPDX-License-Identifier: MIT
// Original work for this crate.

//! The `graph` subcommand: the skeleton and masks that `segment` published, to
//! a vessel graph written as NGFF object tables into the image (and, on
//! request, as JSON), optionally annotated from atlas labels and with a branch
//! density image.
//!
//! The stages, in order:
//!
//! 1. the graph ([`vessel_graph`]): radii from `<name>-mask`, `artery_binary`
//!    and `vein_binary` from the per-channel masks `segment` published
//!    (`<name>-artery-mask`, `<name>-vein-mask`), `artery_raw` / `vein_raw`
//!    from the raw channels when given;
//! 2. atlas coordinates (`coordinates_atlas`): the graph's voxels mapped into
//!    atlas voxels through a [`PointMap`] — the change of lattice onto an
//!    atlas label already in the sample frame (`--annotation-label`), or,
//!    with `--registration`, onto the registration's sample and through its
//!    point transform into a prepared atlas
//!    (`blockflow_atlas_registration::atlas_space`), with the radii as
//!    `_atlas` twins in atlas voxels;
//! 3. annotation ([`annotate`]): region, hemisphere and distance to surface,
//!    read on the atlas grid;
//! 4. classification ([`classify`]) when there is an artery mask;
//! 5. branch density ([`branch_density`]) as a one-level OME-Zarr image, on
//!    the atlas grid when there is an atlas;
//! 6. the tables ([`write_graph_tables`]) and the JSON.
//!
//! Stage 1 is planned over blocks of the stored labels and channels
//! ([`super::planned`]); the atlas volumes, the branch density and the graph
//! itself are resident (see the README's limitations).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use atlas_regions::{
    distance_to_surface, AnnotateParams, Annotator, AtlasVolumes, Ontology, Rounding,
};
use blockflow::ome_zarr::{OmeVolumeLevel, OmeVolumePyramid};
use blockflow::ops::distance::DistanceParams;
use blockflow::voxels::Voxels;
use blockflow::zarr_env::{AttachedImage, PhysicalFrame, PyramidSpec};
use blockflow::{Error, Result, ZarrEnvironment};
use blockflow_atlas_registration::atlas_space::{
    ImageFrame, ImageToSample, LookupParams, OpenParams, RegisteredAtlas,
};
use clap::Args;
use serde_json::Value;

use super::annotate::ATLAS_COORDINATES;
use crate::profile::GRAPH;

use super::planned::{PlanParams, StoredInputs};
use super::tables::{write_graph_tables, GraphTables, TableGeometry};
use super::vessel::Factors;
use super::{
    annotate, branch_density, classify, vessel_graph_stored, Affine, BranchParams, ClassifyParams,
    GraphParams, Lattice, PointMap,
};

/// `graph`: a published skeleton (and masks) to a vessel graph.
#[derive(Debug, Clone, Args)]
pub struct GraphArgs {
    /// OME-Zarr image that `segment` published its labels into.
    #[arg(long)]
    pub zarr: PathBuf,
    /// Label prefix `segment` used: reads `labels/<name>-skeleton`,
    /// `labels/<name>-mask` and, where present, `labels/<name>-artery-mask`
    /// and `labels/<name>-vein-mask`. Also the tables' prefix.
    #[arg(long, default_value = "vessel")]
    pub name: String,
    /// Artery-channel index in the image, read as `artery_raw`.
    #[arg(long)]
    pub artery_channel: Option<usize>,
    /// Artery mask label (default `<name>-artery-mask` when it exists), read as
    /// `artery_binary`; with it the edges are classified as artery and vein.
    #[arg(long)]
    pub artery_mask: Option<String>,
    /// Vein-channel index in the image, read as `vein_raw`.
    #[arg(long)]
    pub vein_channel: Option<usize>,
    /// Vein mask label (default `<name>-vein-mask` when it exists).
    #[arg(long)]
    pub vein_mask: Option<String>,
    /// Keep every connected component rather than only the largest.
    #[arg(long)]
    pub all_components: bool,
    /// Atlas annotation already in the sample frame: a label image of this
    /// name in the same OME-Zarr (any lattice of the image's physical frame).
    #[arg(long)]
    pub annotation_label: Option<String>,
    /// Map the graph into atlas space through this registration (a directory
    /// `atlas-registration register` or `chain` wrote, with the sample as the
    /// first fixed image and the atlas as the last moving one), and annotate,
    /// measure and voxelise there, against the prepared atlas.
    #[arg(long, conflicts_with_all = ["annotation_label", "hemisphere_label", "distance_label"])]
    pub registration: Option<PathBuf>,
    /// The prepared atlas (`atlas-prepare` output) the registration used as
    /// its moving image (default: that moving image, if it is a prepared atlas).
    #[arg(long, requires = "registration")]
    pub atlas: Option<PathBuf>,
    /// Do not compare the atlas's reference with the registration's moving
    /// image voxel for voxel (the shapes are still compared).
    #[arg(long, requires = "registration")]
    pub skip_atlas_check: bool,
    /// Hemisphere label image on the annotation's lattice.
    #[arg(long)]
    pub hemisphere_label: Option<String>,
    /// Distance-to-surface image on the annotation's lattice, used as is
    /// (default: computed from the annotation, in `--distance-unit`s).
    #[arg(long)]
    pub distance_label: Option<String>,
    /// The physical length one unit of the computed distance to surface is:
    /// 25 (µm) gives the 25 µm atlas voxels the classifier's thresholds are in.
    /// Not used with `--registration`, where the distance is in the prepared
    /// atlas's voxels.
    #[arg(long, default_value_t = 25.0)]
    pub distance_unit: f64,
    /// Region tree (Allen-style JSON lines), needed with `--level` (with
    /// `--registration`, the prepared atlas's own tree is used instead).
    #[arg(long, conflicts_with = "registration")]
    pub ontology: Option<PathBuf>,
    /// Collapse every region onto its ancestor at this level: the JSON lines'
    /// `st_level` with `--ontology`, the tree depth (the reference's level)
    /// with `--registration`.
    #[arg(long)]
    pub level: Option<i64>,
    /// Write the branch density as a one-level OME-Zarr image here.
    #[arg(long)]
    pub branch_density: Option<PathBuf>,
    /// The density's lattice: this level of the image's pyramid (default: the
    /// atlas grid with `--registration`, the annotation's lattice with
    /// `--annotation-label`, else the coarsest level).
    #[arg(long)]
    pub density_level: Option<usize>,
    /// Density ball radius, z,y,x voxels of the density's lattice.
    #[arg(long, value_delimiter = ',', default_values_t = vec![15, 15, 15])]
    pub density_radius: Vec<usize>,
    /// Do not write the object tables.
    #[arg(long)]
    pub no_tables: bool,
    /// Also write the graph as JSON here.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Block edge of the planned vertex measure (raw graph, radii, signals),
    /// z,y,x. Does not change any result.
    #[arg(long, value_delimiter = ',', default_values_t = vec![128, 256, 256])]
    pub block: Vec<usize>,
    /// Block edge of the planned narrowing of the published labels, z,y,x.
    /// Does not change any result.
    #[arg(long, value_delimiter = ',', default_values_t = vec![64, 64, 128])]
    pub pack_block: Vec<usize>,
    /// Worker threads (default: one per core).
    #[arg(long)]
    pub workers: Option<usize>,
    /// Chunk cache of the stored volumes, in MiB. Does not change any result.
    #[arg(long, default_value_t = 256)]
    pub cache_mb: u64,
    /// Scratch directory for the planned stages (default: next to the image).
    /// Removed afterwards.
    #[arg(long)]
    pub work: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub vertices: usize,
    pub edges: usize,
    pub classified: bool,
    pub annotated: bool,
    pub tables: Option<GraphTables>,
    /// Vertices outside the branch density's lattice.
    pub density_outside: Option<usize>,
    /// With `--registration`: vertices mapped outside the atlas (region 0).
    pub atlas_outside: Option<usize>,
}

fn label_root(root: &Path, name: &str) -> PathBuf {
    root.join("labels").join(name)
}

/// A label image's level 0 and its lattice.
fn read_label(root: &Path, name: &str) -> Result<(Lattice, Voxels)> {
    let path = label_root(root, name);
    if !path.join("zarr.json").exists() {
        return Err(Error::invalid(format!(
            "the image has no label `{name}` ({} does not exist)",
            path.display()
        )));
    }
    let pyramid = OmeVolumePyramid::open(&path, 0, 0)?;
    let lattice = level_lattice(&pyramid.levels()[0]);
    Ok((lattice, pyramid.attached_level(0)?.read()?))
}

/// A level's `scale` and `translation`, already projected onto `z, y, x`.
pub(crate) fn level_lattice(level: &OmeVolumeLevel) -> Lattice {
    let mut lattice = Lattice::unit();
    let Some(rows) = level.coordinate_transformations.as_array() else {
        return lattice;
    };
    for row in rows {
        let (key, slot) = match row.get("type").and_then(Value::as_str) {
            Some("scale") => ("scale", &mut lattice.scale),
            Some("translation") => ("translation", &mut lattice.translation),
            _ => continue,
        };
        if let Some(vector) = row.get(key).and_then(Value::as_array) {
            for (axis, value) in vector.iter().take(3).enumerate() {
                if let Some(value) = value.as_f64() {
                    slot[axis] = value;
                }
            }
        }
    }
    lattice
}

/// The spatial unit the image declares on `x`, `voxel` if none.
pub(crate) fn image_unit(root: &Path) -> String {
    std::fs::read_to_string(root.join("zarr.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|metadata| {
            let multiscale = metadata
                .pointer("/attributes/ome/multiscales/0")
                .or_else(|| metadata.pointer("/attributes/multiscales/0"))?
                .clone();
            multiscale
                .get("axes")?
                .as_array()?
                .iter()
                .find(|axis| axis.get("name").and_then(Value::as_str) == Some("x"))?
                .get("unit")?
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_else(|| "voxel".to_string())
}

/// The level-0 frame of the image at `root` on `lattice`, with the unit it
/// declares (none for `voxel`).
pub(crate) fn image_frame(root: &Path, lattice: Lattice) -> ImageFrame {
    let unit = image_unit(root);
    let unit = (unit != "voxel").then_some(unit.as_str());
    ImageFrame::new(lattice.scale, lattice.translation, unit)
}

/// The atlas grid's lattice: the prepared atlas's own frame.
pub(crate) fn atlas_lattice(registered: &RegisteredAtlas) -> Lattice {
    Lattice {
        scale: registered.atlas_image.frame.scale,
        translation: registered.atlas_image.frame.translation,
    }
}

/// The conversions from image voxels and µm to atlas voxels the radii take,
/// the reference's: the image-to-sample scale is the resample factor, and a
/// sample voxel is taken as an atlas voxel.
fn radius_factors(
    registered: &RegisteredAtlas,
    to_sample: &ImageToSample,
    image: Lattice,
) -> Factors {
    let sample = registered.sample().frame.scale;
    let mean = |v: [f64; 3]| (v[0] + v[1] + v[2]) / 3.0;
    Factors {
        resample: to_sample.scale,
        atlas_spacing: sample,
        spacing_mean: mean(image.scale),
        atlas_spacing_mean: mean(sample),
    }
}

/// `explicit`, else `<default>` when that label exists, else none.
fn optional_label(root: &Path, explicit: &Option<String>, default: String) -> Option<String> {
    match explicit {
        Some(name) => Some(name.clone()),
        None => label_root(root, &default)
            .join("zarr.json")
            .exists()
            .then_some(default),
    }
}

/// Level 0 of a published label, as a stored array.
fn stored_label(root: &Path, name: &str) -> Result<AttachedImage> {
    let path = label_root(root, name);
    if !path.join("zarr.json").exists() {
        return Err(Error::invalid(format!(
            "the image has no label `{name}` ({} does not exist)",
            path.display()
        )));
    }
    OmeVolumePyramid::open(&path, 0, 0)?.attached_level(0)
}

/// Level 0 of a channel of the image, as a stored array.
fn stored_channel(root: &Path, channel: Option<usize>) -> Result<Option<AttachedImage>> {
    channel
        .map(|channel| OmeVolumePyramid::open(root, channel, 0)?.attached_level(0))
        .transpose()
}

/// The atlas volumes of `--annotation-label` and its companions, all on the
/// annotation's lattice.
struct SampleAtlas {
    lattice: Lattice,
    annotation: Voxels,
    hemispheres: Option<Voxels>,
    distance: Voxels,
}

impl SampleAtlas {
    fn read(root: &Path, args: &GraphArgs, annotation: &str) -> Result<Self> {
        let (lattice, annotation) = read_label(root, annotation)?;
        let shape = annotation.shape();
        let companion = |name: &Option<String>, what: &str| -> Result<Option<Voxels>> {
            let Some(name) = name else {
                return Ok(None);
            };
            let (other, volume) = read_label(root, name)?;
            if volume.shape() != shape || other != lattice {
                return Err(Error::invalid(format!(
                    "the {what} label `{name}` is {:?} on {other:?} and the annotation {shape:?} \
                     on {lattice:?}; both must share the annotation's lattice",
                    volume.shape()
                )));
            }
            Ok(Some(volume))
        };
        let hemispheres = companion(&args.hemisphere_label, "hemisphere")?;
        let distance = match companion(&args.distance_label, "distance")? {
            // A field published as a label is an integer; read it as f64.
            Some(Voxels::U64(field)) => Voxels::F64(field.mapv(|v| v as f64)),
            Some(Voxels::I64(field)) => Voxels::F64(field.mapv(|v| v as f64)),
            Some(field) => field,
            None => {
                if !args.distance_unit.is_finite() || args.distance_unit <= 0.0 {
                    return Err(Error::invalid(format!(
                        "--distance-unit is {}; a length is positive",
                        args.distance_unit
                    )));
                }
                let sampling = lattice.scale.map(|scale| scale / args.distance_unit);
                distance_to_surface(
                    &annotation,
                    &DistanceParams::default().with_sampling(sampling),
                )?
            }
        };
        Ok(Self {
            lattice,
            annotation,
            hemispheres,
            distance,
        })
    }

    fn volumes(&self) -> AtlasVolumes<'_> {
        let volumes = AtlasVolumes::new(&self.annotation).with_distance_to_surface(&self.distance);
        match &self.hemispheres {
            Some(hemispheres) => volumes.with_hemispheres(hemispheres),
            None => volumes,
        }
    }
}

fn annotator(args: &GraphArgs) -> Result<Annotator> {
    let annotator = Annotator::new(AnnotateParams::default().with_level(args.level));
    match &args.ontology {
        Some(path) => Ok(annotator.with_ontology(Arc::new(Ontology::from_jsonl_file(path)?))),
        None if args.level.is_some() => Err(Error::invalid(
            "--level collapses regions on the region tree; give it with --ontology",
        )),
        None => Ok(annotator),
    }
}

/// Write `voxels` (`f64`) as a one-level `float32` OME-Zarr image at `root`,
/// replacing what is there.
pub(crate) fn write_density(
    root: &Path,
    voxels: &Voxels,
    lattice: Lattice,
    unit: &str,
) -> Result<()> {
    if root.exists() {
        std::fs::remove_dir_all(root).map_err(Error::backend)?;
    }
    let narrow = Voxels::F32(voxels.view::<f64>()?.mapv(|value| value as f32));
    let shape = narrow.shape();
    let chunk = shape.map(|edge| edge.clamp(1, 128));
    let frame = PhysicalFrame::new(lattice.scale, Some(unit)).with_origin(lattice.translation);
    let spec = PyramidSpec::new(Vec::new())?
        .with_chunk(chunk)
        .with_physical(frame);
    ZarrEnvironment::build_multiscale(root, &narrow, &spec)?;
    Ok(())
}

pub fn run_graph(args: &GraphArgs) -> Result<GraphSummary> {
    let started = Instant::now();
    let image = OmeVolumePyramid::open(&args.zarr, 0, 0)?;
    let root = image.image_root().to_path_buf();
    let level0 = level_lattice(&image.levels()[0]);
    let shape = image.levels()[0].shape;
    let unit = image_unit(&root);

    // The registration and atlas first, so that a mismatch fails before the
    // graph is built.
    GRAPH.stage("open registration");
    let registered = args
        .registration
        .as_deref()
        .map(|dir| {
            let params = OpenParams {
                atlas: args.atlas.clone(),
                check_reference: !args.skip_atlas_check,
            };
            let registered = RegisteredAtlas::open(dir, &params)?;
            let to_sample = registered.image_to_sample(&image_frame(&root, level0), [0.0; 3])?;
            Ok::<_, Error>((registered, to_sample))
        })
        .transpose()?;
    if let Some((registered, to_sample)) = &registered {
        eprintln!(
            "graph: atlas {} {:?} through {} ({} step(s))",
            registered.atlas.root.display(),
            registered.shape(),
            registered.registration.display(),
            registered.transform.steps()
        );
        if let Some(warning) = registered.frame_warning(to_sample, shape) {
            eprintln!("graph: warning: {warning}");
        }
    }

    // 1. The graph, planned over blocks of the stored labels and channels.
    let artery_label = optional_label(
        &root,
        &args.artery_mask,
        format!("{}-artery-mask", args.name),
    );
    let vein_label = optional_label(&root, &args.vein_mask, format!("{}-vein-mask", args.name));
    if let Some(name) = &artery_label {
        eprintln!("graph: artery mask `{name}`");
    }
    let inputs = StoredInputs {
        skeleton: stored_label(&root, &format!("{}-skeleton", args.name))?,
        vessels: stored_label(&root, &format!("{}-mask", args.name))?,
        artery_intensity: stored_channel(&root, args.artery_channel)?,
        artery_mask: artery_label
            .as_deref()
            .map(|name| stored_label(&root, name))
            .transpose()?,
        vein_intensity: stored_channel(&root, args.vein_channel)?,
        vein_mask: vein_label
            .as_deref()
            .map(|name| stored_label(&root, name))
            .transpose()?,
        spacing: image.voxel_size(),
    };
    let params = GraphParams {
        largest_component: !args.all_components,
        ..GraphParams::default()
    };
    let plan = PlanParams {
        block: args
            .block
            .as_slice()
            .try_into()
            .map_err(|_| Error::invalid("--block needs three values, z,y,x"))?,
        pack_block: args
            .pack_block
            .as_slice()
            .try_into()
            .map_err(|_| Error::invalid("--pack-block needs three values, z,y,x"))?,
        workers: args
            .workers
            .unwrap_or_else(rayon::current_num_threads)
            .max(1),
        cache_bytes: args.cache_mb.max(1) << 20,
        work: args.work.clone().unwrap_or_else(|| {
            let name = args
                .zarr
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "image".to_string());
            args.zarr
                .with_file_name(format!(".{name}.vessel-graph-graph-work"))
        }),
    };
    let mut graph = vessel_graph_stored(&inputs, &params, &plan)?;
    eprintln!(
        "graph: {} vertices, {} edges ({:.1}s)",
        graph.n_vertices(),
        graph.n_edges(),
        started.elapsed().as_secs_f64()
    );

    // 2-3. Atlas coordinates and annotation.
    GRAPH.stage("atlas: map and annotate");
    let atlas = args
        .annotation_label
        .as_deref()
        .map(|name| SampleAtlas::read(&root, args, name))
        .transpose()?;
    if args.annotation_label.is_none()
        && (args.hemisphere_label.is_some() || args.distance_label.is_some())
    {
        return Err(Error::invalid(
            "--hemisphere-label and --distance-label are read on the annotation's lattice; \
             give --annotation-label too",
        ));
    }
    if let Some(atlas) = &atlas {
        // The atlas is already in the sample frame, so the map from graph
        // voxels to atlas voxels is the change of lattice.
        let to_atlas: Box<dyn PointMap> = Box::new(Affine::between(level0, atlas.lattice)?);
        graph.transform_coordinates(ATLAS_COORDINATES, to_atlas.as_ref())?;
        annotate(&mut graph, &annotator(args)?, atlas.volumes())?;
    }
    let mut atlas_outside = None;
    if let Some((registered, to_sample)) = &registered {
        // The atlas in its own space: graph voxels onto the registration's
        // sample lattice, then through its point transform.
        let to_atlas = |points: &[[f64; 3]]| -> Result<Vec<[f64; 3]>> {
            Ok(registered.map(to_sample, points).atlas)
        };
        graph.transform_coordinates(ATLAS_COORDINATES, &to_atlas)?;
        graph.scale_radii_with(&radius_factors(registered, to_sample, level0))?;
        let lookup = LookupParams {
            level: args.level,
            rounding: Rounding::Nearest,
        };
        annotate(
            &mut graph,
            &registered.annotator(&lookup),
            registered.volumes(),
        )?;
        let outside = registered.outside(&graph.points(ATLAS_COORDINATES)?, Rounding::Nearest);
        eprintln!(
            "graph: {outside} of {} vertices map outside the atlas",
            graph.n_vertices()
        );
        atlas_outside = Some(outside);
    }

    // 4. Classification.
    GRAPH.stage("classify");
    let classified = graph.edge("artery_binary").is_some();
    if classified {
        let trace = classify(&mut graph, &ClassifyParams::default())?;
        eprintln!(
            "graph: {} artery and {} vein edges",
            trace.arteries(),
            trace.veins()
        );
    }

    // 5. Branch density.
    GRAPH.stage("branch density");
    let mut density_outside = None;
    if let Some(path) = &args.branch_density {
        let radius: [usize; 3] = args
            .density_radius
            .as_slice()
            .try_into()
            .map_err(|_| Error::invalid("--density-radius needs three values, z,y,x"))?;
        let atlas_grid = match (&atlas, &registered) {
            (Some(atlas), _) => Some((atlas.lattice, atlas.annotation.shape())),
            (None, Some((registered, _))) => Some((atlas_lattice(registered), registered.shape())),
            (None, None) => None,
        };
        let (lattice, density_shape, column) = match (atlas_grid, args.density_level) {
            (Some((lattice, shape)), None) => (lattice, shape, ATLAS_COORDINATES),
            (_, level) => {
                let levels = image.levels();
                let level = level.unwrap_or(levels.len() - 1);
                let chosen = levels.get(level).ok_or_else(|| {
                    Error::invalid(format!(
                        "--density-level {level}; the image has {} levels",
                        levels.len()
                    ))
                })?;
                let lattice = level_lattice(chosen);
                graph.transform_coordinates(
                    "coordinates_density",
                    &Affine::between(level0, lattice)?,
                )?;
                (lattice, chosen.shape, "coordinates_density")
            }
        };
        let params = BranchParams {
            coordinates: column.to_string(),
            radius,
            ..BranchParams::default()
        };
        let (density, outside) = branch_density(&graph, density_shape, &params)?;
        write_density(path, &density, lattice, &unit)?;
        density_outside = Some(outside);
    }

    // 6. Output.
    GRAPH.stage("tables and json");
    let tables = match args.no_tables {
        true => None,
        false => {
            let geometry = TableGeometry {
                shape,
                lattice: level0,
                unit,
                atlas: registered
                    .as_ref()
                    .map(|(registered, _)| registered.describe()),
            };
            Some(write_graph_tables(
                &graph,
                &root.join("tables"),
                &args.name,
                &geometry,
            )?)
        }
    };
    if let Some(path) = &args.out {
        graph.write_json(path)?;
    }
    GRAPH.finish();
    eprintln!("graph: done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(GraphSummary {
        vertices: graph.n_vertices(),
        edges: graph.n_edges(),
        classified,
        annotated: atlas.is_some() || registered.is_some(),
        tables,
        density_outside,
        atlas_outside,
    })
}
