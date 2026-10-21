// SPDX-License-Identifier: MIT
// Original work for this crate.

//! LFA-1 puncta in T-cell immune synapses on supported lipid bilayers.
//!
//! One TIRF plane per OME-Zarr image (`[t, c, z, y, x]`, one channel: LFA-1,
//! m24-AF488, 65 nm pixels). Two planned branches over the same input, then
//! two measurement plans:
//!
//! | step | blockflow |
//! |---|---|
//! | spot response: difference of Gaussians, `sigma` and `1.6 sigma` | `ops::interest::BlobResponseOp` |
//! | peak level: `median + k * 1.4826 MAD` of the response | `SampledValuesOp`, `ApplyGlobalThresholdOp` (`RobustSigma`, `KeepAtOrAbove`) |
//! | peaks: 8-connected regional maxima of that | `append_response_peak_table_phases` |
//! | greedy suppression, strongest first, of peaks closer than `min_separation` | `SuppressRowsOp` |
//! | one seed per kept peak | [`seeds::PeakSeedsOp`] (here: no library op takes id-less rows) |
//! | spot extent: response above `median + extent * k * sigma` | `ApplyGlobalThresholdOp` (`RobustSigma`, mask) |
//! | spot labels: seeded watershed of the negated response inside the extent | `VoxelwiseMapOp`, `SeededWatershedOp` (`Adjacent`) |
//! | synapse mask: Gaussian, then a level above the histogram-mode background | `SmoothOp`, `ApplyGlobalThresholdOp` (`BackgroundMode`) |
//! | close, fill the cleared centre, label | `MorphologyOp` (disk), `ops::fill`, `LabelComponentsOp` |
//! | per spot: shape, raw and response intensity, the synapse label under it | `Measurements` over the spot labels |
//! | per synapse component: shape and raw intensity | `Measurements` over the synapse labels |
//!
//! | distance to the synapse edge | `ops::distance` (exact EDT), measured per spot |
//! | both label images as `uint64`, for publication | `WidenLabelsOp` |
//!
//! The host then picks the synapse (largest component, discounted by distance
//! from the field centre) and installs the segmentation in the input image as
//! OME-Zarr labels with object tables ([`publish`]). The CSVs under `--out`
//! gather the same rows over all images for the cross-image analysis. The
//! Python reference this follows is `scripts/spots_reference.py`. Per-synapse
//! features for the cross-condition analysis are in [`features`].

#[path = "../../support/planning.rs"]
mod example_planning;
mod features;
mod publish;
mod seeds;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::geometry::BlockGrid;
use blockflow::op::{Chain, SourceInput};
use blockflow::ops::label::{LabelComponentsOp, RelabelComponentsOp};
use blockflow::ops::measure::{
    collect_class_a_shapes, collect_class_a_values, IntensityImage, IntensityMeasurements,
    IntensitySet, Measurements, ShapeMeasurements, ShapeSet,
};
use blockflow::ops::distance::{self, DistanceParams};
use blockflow::ops::{
    append_response_peak_table_phases, fill, ApplyGlobalThresholdOp, BlobResponseOp, Boundary,
    Connectivity, ElementShape, Gaussian, GlobalThreshold, GlobalThresholdOutput, Morphology,
    MorphologyOp, RadiusPair, RowStreams, SampledValuesOp, SeededWatershedOp, Separation,
    SmoothOp, StructuringElement, SuppressRowsOp, Suppression, SuppressionRadius, ThresholdTest,
    VoxelwiseMapOp, VoxelwiseMaskOp, WidenLabelsOp,
};
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{Error, Result};
use clap::Parser;
use ndarray::{Array2, Axis};

/// Gaussian kernels end at four sigma: SciPy's default, which the reference uses.
const TRUNCATE: f64 = 4.0;

#[derive(Debug, Parser)]
#[command(
    name = "lewis-synapse-spots",
    about = "Detect LFA-1 puncta and the synapse outline in single-plane OME-Zarr images"
)]
struct Config {
    /// OME-Zarr images, or directories searched (one level) for `*.ome.zarr`.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long, default_value = ".tmp/lewis-synapse-spots")]
    out: PathBuf,
    /// Spot Gaussian sigma, pixels; the wider DoG Gaussian is `1.6` times it.
    #[arg(long, default_value_t = 1.5)]
    spot_sigma: f64,
    #[arg(long, default_value_t = 1.6)]
    spot_sigma_ratio: f64,
    /// Peak level, robust sigmas of the DoG response above its median.
    #[arg(long, default_value_t = 5.0)]
    spot_k: f64,
    /// Spot extent level as a fraction of `spot_k`.
    #[arg(long, default_value_t = 0.5)]
    spot_extent: f64,
    /// Peaks closer than this (pixels) to a stronger kept peak are dropped.
    #[arg(long, default_value_t = 2.5)]
    min_separation: f64,
    #[arg(long, default_value_t = 6.0)]
    cell_sigma: f64,
    /// Synapse level, background noise sigmas above the mode...
    #[arg(long, default_value_t = 5.0)]
    cell_k: f64,
    /// ...or this fraction of `P_upper - mode`, whichever is larger.
    #[arg(long, default_value_t = 0.15)]
    cell_fraction: f64,
    #[arg(long, default_value_t = 99.5)]
    cell_upper: f64,
    #[arg(long, default_value_t = 512)]
    cell_bins: usize,
    /// Disk radius of the closing that bridges the gaps between puncta.
    #[arg(long, default_value_t = 10)]
    close_radius: usize,
    /// Published as `labels/<layer>-spots` and `labels/<layer>-synapse`, with
    /// tables of the same names, in each input image.
    #[arg(long, default_value = "lewis")]
    layer: String,
    /// Replace an existing publication of `--layer`.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

/// The images a run keeps, by role.
struct Plan {
    base: blockflow::assemble::Assembly,
    spots: blockflow::ops::measure::MeasurementPlan,
    cells: blockflow::ops::measure::MeasurementPlan,
    spot_labels: ImageId,
    cell_labels: ImageId,
    edge_distance: ImageId,
    /// The label images widened to `uint64`, as published.
    spot_labels_u64: ImageId,
    cell_labels_u64: ImageId,
}

#[derive(Debug, Clone)]
struct CellRow {
    label: u64,
    area: u64,
    centroid: [f64; 2],
    intensity_sum: f64,
    intensity_mean: f64,
    eccentricity: f64,
}

#[derive(Debug, Clone)]
struct SpotRow {
    label: u64,
    area: u64,
    centroid: [f64; 2],
    weighted_centroid: [f64; 2],
    raw_sum: f64,
    raw_max: f64,
    response_max: f64,
    eccentricity: f64,
    major_axis: f64,
    minor_axis: f64,
    cell_label: u64,
    edge_distance: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse();
    let images = discover(&config.inputs)?;
    if images.is_empty() {
        return Err(Error::invalid("lewis-synapse-spots: no *.ome.zarr inputs found"));
    }
    if config.layer.is_empty() || config.layer.contains(['/', '\\']) || config.layer.starts_with('.') {
        return Err(Error::invalid("lewis-synapse-spots: --layer must be one plain path component"));
    }
    fs::create_dir_all(&config.out).map_err(Error::backend)?;
    let mut spots_out = csv(
        &config.out.join("spots.csv"),
        "image,spot,y,x,wy,wx,area,raw_sum,raw_max,response_max,eccentricity,major_axis,minor_axis,cell_label,in_synapse",
    )?;
    let mut images_out = csv(
        &config.out.join("images.csv"),
        "image,height,width,synapse_label,synapse_area,synapse_cy,synapse_cx,synapse_eccentricity,synapse_raw_mean,synapse_raw_sum,components,spots,spots_in_synapse,seconds",
    )?;
    let mut features_out = csv(
        &config.out.join("synapse_features.csv"),
        &format!("image,condition,field,roi,{}", features::header()),
    )?;
    let started = Instant::now();
    for path in &images {
        let name = image_name(path);
        let t0 = Instant::now();
        let pyramid = OmeVolumePyramid::open(path, 0, 0)?;
        let Processed {
            height,
            width,
            cells,
            spots,
            planes,
            work,
            spot_level,
            cell_level,
        } = process(&config, &pyramid)?;
        let synapse = main_component(&cells, [height as f64, width as f64]);
        let synapse_label = synapse.map_or(0, |c| c.label);
        let mut in_synapse = 0usize;
        for spot in &spots {
            let inside = synapse_label != 0 && spot.cell_label == synapse_label;
            in_synapse += usize::from(inside);
            writeln!(
                spots_out,
                "{name},{},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.6},{:.4},{:.4},{:.4},{},{}",
                spot.label,
                spot.centroid[0],
                spot.centroid[1],
                spot.weighted_centroid[0],
                spot.weighted_centroid[1],
                spot.area,
                spot.raw_sum,
                spot.raw_max,
                spot.response_max,
                spot.eccentricity,
                spot.major_axis,
                spot.minor_axis,
                spot.cell_label,
                u8::from(inside)
            )
            .map_err(Error::backend)?;
        }
        let mut synapse_features = None;
        if let Some(c) = synapse {
            let synapse_mask = planes.cell_labels.mapv(|l| u64::from(l) == c.label);
            let feature_spots: Vec<features::Spot> = spots
                .iter()
                .filter(|s| s.cell_label == c.label)
                .map(|s| features::Spot {
                    y: s.centroid[0],
                    x: s.centroid[1],
                    area: s.area as f64,
                    raw_sum: s.raw_sum,
                    ecc: s.eccentricity,
                    major_axis: s.major_axis,
                    minor_axis: s.minor_axis,
                    edge_distance: s.edge_distance,
                })
                .collect();
            let spot_pixels = planes.spot_labels.mapv(|l| l != 0);
            let distance = ndarray::Zip::from(&planes.edge_distance)
                .and(&synapse_mask)
                .map_collect(|&d, &m| if m { d } else { 0.0 });
            let values = features::compute(&features::Inputs {
                raw: planes.raw.view(),
                synapse: synapse_mask.view(),
                distance: distance.view(),
                spot_pixels: spot_pixels.view(),
                spots: &feature_spots,
                eccentricity: c.eccentricity,
                pixel_um: pyramid.voxel_size()[1],
                seed: name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
                }),
            });
            let (condition, field, roi) = identity(&name);
            let row = features::row(&values);
            writeln!(features_out, "{name},{condition},{field},{roi},{row}").map_err(Error::backend)?;
            synapse_features = Some(values);
        }
        let volume = [1, height, width];
        let (spots_label, synapse_label_name) =
            (publish::spots_name(&config.layer), publish::synapse_name(&config.layer));
        let tables = work.join("tables");
        publish::spot_table(&tables.join(&spots_label), volume, &spots_label, &spots, synapse_label)?;
        publish::synapse_table(
            &tables.join(&synapse_label_name),
            volume,
            &synapse_label_name,
            synapse.zip(synapse_features.as_deref()),
        )?;
        publish::install(&pyramid, &spots_label, &spot_level, &tables.join(&spots_label), config.overwrite)?;
        publish::install(
            &pyramid,
            &synapse_label_name,
            &cell_level,
            &tables.join(&synapse_label_name),
            config.overwrite,
        )?;
        fs::remove_dir_all(&work).map_err(Error::backend)?;
        let (condition, field, roi) = identity(&name);
        publish::image_metadata(&pyramid, &condition, &field, &roi)?;
        let seconds = t0.elapsed().as_secs_f64();
        match synapse {
            Some(c) => writeln!(
                images_out,
                "{name},{height},{width},{},{},{:.3},{:.3},{:.4},{:.3},{:.1},{},{},{in_synapse},{seconds:.3}",
                c.label,
                c.area,
                c.centroid[0],
                c.centroid[1],
                c.eccentricity,
                c.intensity_mean,
                c.intensity_sum,
                cells.len(),
                spots.len()
            ),
            None => writeln!(
                images_out,
                "{name},{height},{width},0,0,,,,,,0,{},0,{seconds:.3}",
                spots.len()
            ),
        }
        .map_err(Error::backend)?;
        println!(
            "{name}: {} spots, {in_synapse} in synapse ({} px), {seconds:.2}s",
            spots.len(),
            synapse.map_or(0, |c| c.area)
        );
    }
    spots_out.flush().map_err(Error::backend)?;
    features_out.flush().map_err(Error::backend)?;
    images_out.flush().map_err(Error::backend)?;
    write_params(&config, images.len(), started.elapsed().as_secs_f64())?;
    Ok(())
}

/// One image's results, with its work directory still holding the `uint64`
/// label arrays to publish.
struct Processed {
    height: usize,
    width: usize,
    cells: Vec<CellRow>,
    spots: Vec<SpotRow>,
    planes: Planes,
    work: PathBuf,
    spot_level: PathBuf,
    cell_level: PathBuf,
}

fn process(config: &Config, pyramid: &OmeVolumePyramid) -> Result<Processed> {
    let image_root = pyramid.image_root();
    for label in [publish::spots_name(&config.layer), publish::synapse_name(&config.layer)] {
        for dir in ["labels", "tables"] {
            let existing = image_root.join(dir).join(&label);
            if existing.exists() && !config.overwrite {
                return Err(Error::invalid(format!(
                    "lewis-synapse-spots: {} exists; pass --overwrite to replace it",
                    existing.display()
                )));
            }
        }
    }
    // level 0, the one plane at t = 0, c = 0
    let source = pyramid.attached_level(0)?;
    let (dtype, volume) = source.metadata()?;
    if volume[0] != 1 {
        return Err(Error::invalid(format!(
            "lewis-synapse-spots: {} has {} z planes; this example is single-plane",
            image_root.display(),
            volume[0]
        )));
    }
    let plan = build(config, volume, dtype)?;
    // next to its destination, so publishing moves rather than copies
    let work = image_root
        .join("labels")
        .join(format!(".{}-blockflow-work", config.layer));
    if work.exists() {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&source))?;
    let mut phase_work = plan.base.work();
    phase_work.extend(plan.spots.phase_work());
    phase_work.extend(plan.cells.phase_work());
    let hints = Hints {
        keep_images: [
            plan.spot_labels,
            plan.cell_labels,
            plan.edge_distance,
            plan.spot_labels_u64,
            plan.cell_labels_u64,
        ]
        .into_iter()
        .collect(),
        ..Hints::default()
    };
    execute_phases(
        "lewis synapse spots",
        &plan.base.workflow,
        &plan.cells.decomposition,
        &hints,
        &env,
        &[],
        &phase_work,
    )?;

    let cells = cell_rows(&env, &plan.cells, volume)?;
    let spots = spot_rows(&env, &plan.spots, volume)?;
    let planes = Planes {
        raw: plane_f64(&source.read()?)?,
        cell_labels: env.image(plan.cell_labels.index())?.view::<u32>()?.index_axis(Axis(0), 0).to_owned(),
        spot_labels: env.image(plan.spot_labels.index())?.view::<u32>()?.index_axis(Axis(0), 0).to_owned(),
        edge_distance: env.image(plan.edge_distance.index())?.view::<f64>()?.index_axis(Axis(0), 0).to_owned(),
    };
    drop(env);
    let level = |image: ImageId| work.join(format!("level{}", image.index()));
    Ok(Processed {
        height: volume[1],
        width: volume[2],
        cells,
        spots,
        planes,
        spot_level: level(plan.spot_labels_u64),
        cell_level: level(plan.cell_labels_u64),
        work,
    })
}

/// The planes the per-synapse features read, on the host.
struct Planes {
    raw: Array2<f64>,
    cell_labels: Array2<u32>,
    spot_labels: Array2<u32>,
    edge_distance: Array2<f64>,
}

fn plane_f64(voxels: &blockflow::voxels::Voxels) -> Result<Array2<f64>> {
    let view = voxels.view::<u16>()?;
    Ok(view.index_axis(Axis(0), 0).mapv(f64::from))
}

fn build(config: &Config, volume: [usize; 3], dtype: Dtype) -> Result<Plan> {
    let input = ImageId::from(0);
    // the images are small; one block holds a whole plane
    let grid = BlockGrid::new(volume, volume)?;
    let mut b = PlanBuilder::new(volume, dtype, grid);
    let low = [0.0, config.spot_sigma, config.spot_sigma];
    let high = low.map(|s| s * config.spot_sigma_ratio);

    // ---- spots
    let dog = b
        .pixels(Chain::sequence(vec![
            Chain::source(input, dtype),
            Chain::op(BlobResponseOp::difference_of_gaussians(
                "LFA-1 spot DoG response",
                low,
                high,
                TRUNCATE,
                Boundary::Reflect,
            )?),
        ]))?
        .image()?;
    let response = b.fragments(SampledValuesOp::new(
        "spot response samples",
        "lewis.response",
        SourceInput::voxelwise(dog),
    ))?;
    b.fragments(
        ApplyGlobalThresholdOp::new(
            "spot peak level",
            "lewis.response",
            response.index(),
            GlobalThreshold::robust_sigma(config.spot_k)?,
            GlobalThresholdOutput::KeepAtOrAbove,
        )
        .reading_image(dog.index()),
    )?;
    let (peaks, peak_schema) = append_response_peak_table_phases(
        &mut b,
        "lewis.peaks",
        Lifecycle::DeleteOnExit,
        Connectivity::FacesEdgesAndCorners,
        f64::MIN,
    )?;
    let suppress = SuppressRowsOp::new(
        "peak suppression",
        RowStreams::new(
            "lewis.peaks",
            peaks.index(),
            "lewis.kept",
            Lifecycle::DeleteOnExit,
            peak_schema,
        )?,
        Suppression {
            priority: "response".into(),
            radius: SuppressionRadius::Fixed(config.min_separation),
            pair: RadiusPair::Smaller,
            factor: 1.0,
            position: None,
            voxel_size: [1.0; 3],
        },
    )?;
    let kept_schema = suppress.schema().clone();
    let kept = b.fragments(suppress)?;
    let seeds = b
        .fragments(seeds::PeakSeedsOp::new(
            "peak seeds",
            "lewis.kept",
            kept.index(),
            kept_schema,
        ))?
        .image()?;
    let extent = b
        .fragments(
            ApplyGlobalThresholdOp::new(
                "spot extent level",
                "lewis.response",
                response.index(),
                GlobalThreshold::robust_sigma(config.spot_extent * config.spot_k)?,
                GlobalThresholdOutput::Mask {
                    test: ThresholdTest::Above,
                },
            )
            .reading_image(dog.index()),
        )?
        .image()?;
    b.pixels(Chain::sequence(vec![
        Chain::source(dog, Dtype::F64),
        Chain::op(VoxelwiseMapOp::new("negated spot response", |v| -v)),
    ]))?;
    let spot_labels = b
        .pixels(Chain::op(
            SeededWatershedOp::new("spot basins", seeds, Separation::Adjacent).within(extent),
        ))?
        .image()?;

    // ---- synapse
    let smooth = b
        .pixels(Chain::sequence(vec![
            Chain::source(input, dtype),
            Chain::op(SmoothOp::new(
                "synapse smoothing",
                Gaussian::new([0.0, config.cell_sigma, config.cell_sigma], TRUNCATE)?
                    .with_boundary(Boundary::Reflect),
            )),
        ]))?
        .image()?;
    let smooth_samples = b.fragments(SampledValuesOp::new(
        "synapse samples",
        "lewis.smooth",
        SourceInput::voxelwise(smooth),
    ))?;
    b.fragments(
        ApplyGlobalThresholdOp::new(
            "synapse level",
            "lewis.smooth",
            smooth_samples.index(),
            GlobalThreshold::background_mode(
                config.cell_bins,
                config.cell_k,
                config.cell_fraction,
                config.cell_upper,
            )?,
            GlobalThresholdOutput::Mask {
                test: ThresholdTest::Above,
            },
        )
        .reading_image(smooth.index()),
    )?;
    let r = config.close_radius;
    example_planning::pixels(
        &mut b,
        Chain::op(MorphologyOp::new(
            "synapse closing",
            Morphology::Close,
            StructuringElement::from_radius(ElementShape::Ellipsoid, [0, r, r]),
        )),
    )?;
    fill::append_to(&mut b, "lewis.fill", Lifecycle::DeleteOnExit, Dtype::Bool)?;
    // distance to the synapse edge inside it, for size-normalised radial position
    let side = volume[1].max(volume[2]);
    let edge_distance = distance::append_to(&mut b, &DistanceParams::default(), side)?.image()?;
    // the filled mask again (distance > 0), as the input the labelling reads
    b.pixels(Chain::sequence(vec![
        Chain::source(edge_distance, Dtype::F64),
        Chain::op(VoxelwiseMaskOp::threshold("synapse mask", 0.0)),
    ]))?;
    let cell_labels = label_components(&mut b, "lewis.cells", Connectivity::FacesEdgesAndCorners)?;

    // ---- published copies
    let widen = |b: &mut PlanBuilder, labels: ImageId| -> Result<ImageId> {
        b.pixels(Chain::sequence(vec![
            Chain::source(labels, Dtype::U32),
            Chain::op(WidenLabelsOp),
        ]))?
        .image()
    };
    let spot_labels_u64 = widen(&mut b, spot_labels)?;
    let cell_labels_u64 = widen(&mut b, cell_labels)?;

    let base = b.finish()?;
    let spots = Measurements::for_labels(spot_labels)
        .shape(ShapeSet::standard())
        .intensity(
            IntensityImage::<0>::new(input).holding(dtype),
            IntensitySet::standard(),
        )
        .intensity(
            IntensityImage::<1>::new(dog).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .intensity(
            IntensityImage::<2>::new(cell_labels).holding(Dtype::U32),
            IntensitySet::standard(),
        )
        .intensity(
            IntensityImage::<3>::new(edge_distance).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .stream("lewis.spots")
        .lifecycle(Lifecycle::DeleteOnExit)
        .build(base.decomposition.clone())?;
    let cells = Measurements::for_labels(cell_labels)
        .shape(ShapeSet::standard())
        .intensity(
            IntensityImage::<0>::new(input).holding(dtype),
            IntensitySet::standard(),
        )
        .stream("lewis.cells.rows")
        .lifecycle(Lifecycle::DeleteOnExit)
        .build(spots.decomposition.clone())?;
    Ok(Plan {
        base,
        spots,
        cells,
        spot_labels,
        cell_labels,
        edge_distance,
        spot_labels_u64,
        cell_labels_u64,
    })
}

/// Label the latest (boolean) image's components; returns the label image.
fn label_components(
    b: &mut PlanBuilder,
    stream: &'static str,
    connectivity: Connectivity,
) -> Result<ImageId> {
    let faces = b.fragments(
        LabelComponentsOp::new("component labelling", stream, Lifecycle::DeleteOnExit)
            .connecting(connectivity),
    )?;
    let grid = b.grid().clone();
    b.fragments(
        RelabelComponentsOp::reading("component relabel", stream, faces, &grid)
            .connecting(connectivity),
    )?
    .image()
}

fn intensity_by_label(
    env: &ZarrEnvironment,
    plan: &blockflow::ops::measure::MeasurementPlan,
    index: usize,
    volume: [usize; 3],
) -> Result<BTreeMap<u64, IntensityMeasurements>> {
    let rows = plan
        .class_a_intensity_rows(index)
        .ok_or_else(|| Error::invalid(format!("lewis-synapse-spots: no intensity rows {index}")))?;
    Ok(collect_class_a_values(env, &rows, volume, plan.fixed)?
        .into_iter()
        .map(|values| {
            let m = IntensityMeasurements::from_values(&values);
            (m.label, m)
        })
        .collect())
}

fn shapes(
    env: &ZarrEnvironment,
    plan: &blockflow::ops::measure::MeasurementPlan,
    volume: [usize; 3],
) -> Result<Vec<ShapeMeasurements>> {
    let rows = plan
        .class_a_rows()
        .ok_or_else(|| Error::invalid("lewis-synapse-spots: no shape rows"))?;
    Ok(collect_class_a_shapes(env, &rows, volume, plan.fixed)?
        .iter()
        .map(ShapeMeasurements::from_shape)
        .collect())
}

fn spot_rows(
    env: &ZarrEnvironment,
    plan: &blockflow::ops::measure::MeasurementPlan,
    volume: [usize; 3],
) -> Result<Vec<SpotRow>> {
    let raw = intensity_by_label(env, plan, 0, volume)?;
    let response = intensity_by_label(env, plan, 1, volume)?;
    let cell = intensity_by_label(env, plan, 2, volume)?;
    let distance = intensity_by_label(env, plan, 3, volume)?;
    let mut rows = Vec::new();
    for shape in shapes(env, plan, volume)? {
        let (Some(raw), Some(response), Some(cell)) = (
            raw.get(&shape.label),
            response.get(&shape.label),
            cell.get(&shape.label),
        ) else {
            return Err(Error::invalid(format!(
                "lewis-synapse-spots: spot {} lacks an intensity row",
                shape.label
            )));
        };
        let centroid = shape.centroid.unwrap_or([f64::NAN; 3]);
        let weighted = raw.weighted_centroid.unwrap_or([f64::NAN; 3]);
        let axes = shape.principal_axis_lengths.unwrap_or([f64::NAN; 3]);
        rows.push(SpotRow {
            label: shape.label,
            area: shape.count,
            centroid: [centroid[1], centroid[2]],
            weighted_centroid: [weighted[1], weighted[2]],
            raw_sum: raw.sum,
            raw_max: raw.max.unwrap_or(f64::NAN),
            response_max: response.max.unwrap_or(f64::NAN),
            eccentricity: shape.eccentricity_yx.unwrap_or(f64::NAN),
            major_axis: axes[0],
            minor_axis: axes[1],
            // the synapse component under the spot; the larger label where a
            // spot straddles two, 0 outside every component
            cell_label: cell.max.unwrap_or(0.0) as u64,
            edge_distance: distance
                .get(&shape.label)
                .and_then(|m| m.mean)
                .unwrap_or(f64::NAN),
        });
    }
    rows.sort_by_key(|r| r.label);
    Ok(rows)
}

fn cell_rows(
    env: &ZarrEnvironment,
    plan: &blockflow::ops::measure::MeasurementPlan,
    volume: [usize; 3],
) -> Result<Vec<CellRow>> {
    let raw = intensity_by_label(env, plan, 0, volume)?;
    let mut rows = Vec::new();
    for shape in shapes(env, plan, volume)? {
        let centroid = shape.centroid.unwrap_or([f64::NAN; 3]);
        let intensity = raw.get(&shape.label);
        rows.push(CellRow {
            label: shape.label,
            area: shape.count,
            centroid: [centroid[1], centroid[2]],
            intensity_sum: intensity.map_or(f64::NAN, |m| m.sum),
            intensity_mean: intensity.and_then(|m| m.mean).unwrap_or(f64::NAN),
            eccentricity: shape.eccentricity_yx.unwrap_or(f64::NAN),
        });
    }
    rows.sort_by_key(|r| r.label);
    Ok(rows)
}

/// The synapse: the largest component, discounted by its distance from the
/// field centre (`area / (1 + distance / 50 px)`), since the imaged cell is
/// centred and a neighbour entering from the edge is not.
fn main_component(cells: &[CellRow], shape: [f64; 2]) -> Option<&CellRow> {
    let centre = [shape[0] / 2.0, shape[1] / 2.0];
    let score = |c: &CellRow| {
        let d = (c.centroid[0] - centre[0]).hypot(c.centroid[1] - centre[1]);
        c.area as f64 / (1.0 + d / 50.0)
    };
    cells
        .iter()
        .max_by(|a, b| score(a).total_cmp(&score(b)))
}

/// `20min_icam1ucht1slb_15_m24af488_roi02` -> (`icam1ucht1slb`, `15`, `roi02`).
fn identity(name: &str) -> (String, String, String) {
    let parts: Vec<&str> = name.split('_').collect();
    let get = |i: usize| parts.get(i).copied().unwrap_or("").to_string();
    (get(1), get(2), parts.last().copied().unwrap_or("").to_string())
}

fn discover(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for input in inputs {
        if is_ome_zarr(input) {
            out.push(input.clone());
            continue;
        }
        let mut found: Vec<PathBuf> = fs::read_dir(input)
            .map_err(|err| Error::invalid(format!("lewis-synapse-spots: read {}: {err}", input.display())))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| is_ome_zarr(p))
            .collect();
        found.sort();
        out.extend(found);
    }
    Ok(out)
}

fn is_ome_zarr(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "zarr") && path.join("0").is_dir()
}

fn image_name(path: &Path) -> String {
    let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    name.strip_suffix(".ome.zarr")
        .or_else(|| name.strip_suffix(".zarr"))
        .unwrap_or(&name)
        .to_string()
}

fn csv(path: &Path, header: &str) -> Result<BufWriter<File>> {
    let mut out = BufWriter::new(File::create(path).map_err(Error::backend)?);
    writeln!(out, "{header}").map_err(Error::backend)?;
    Ok(out)
}

fn write_params(config: &Config, images: usize, seconds: f64) -> Result<()> {
    let params = serde_json::json!({
        "images": images,
        "seconds": seconds,
        "spot_sigma": config.spot_sigma,
        "spot_sigma_ratio": config.spot_sigma_ratio,
        "spot_k": config.spot_k,
        "spot_extent": config.spot_extent,
        "min_separation": config.min_separation,
        "cell_sigma": config.cell_sigma,
        "cell_k": config.cell_k,
        "cell_fraction": config.cell_fraction,
        "cell_upper": config.cell_upper,
        "cell_bins": config.cell_bins,
        "close_radius": config.close_radius,
    });
    fs::write(
        config.out.join("params.json"),
        serde_json::to_string_pretty(&params).expect("params serialize") + "\n",
    )
    .map_err(Error::backend)
}
