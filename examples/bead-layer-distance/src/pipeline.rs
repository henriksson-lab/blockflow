// SPDX-License-Identifier: MIT
//
// Original work for this crate.

//! The whole analysis as one blockflow plan over level 0 (image 0 = AF488,
//! supplied 0 = AF594, supplied 1 = C-Blue), then publishing, CSVs and QC.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::fragment::fold_fragments;
use blockflow::object_table::finalize_object_table;
use blockflow::ome_zarr::OmeVolumePyramid;
use blockflow::ome_zarr::{publish_label, LabelPublication};
use blockflow::op::SourceInput;
use blockflow::ops::label::{LabelComponentsOp, RelabelComponentsOp};
use blockflow::ops::rows::collect_rows;
use blockflow::ops::surfaces::{
    append_layered_surface_phases, LayeredSurfaces, SurfaceBandOp, SurfaceColumns, SurfaceCue,
    SurfaceLayer, SurfaceLinesOp,
};
use blockflow::ops::{
    append_response_peak_table_phases, ApplyGlobalThresholdOp, BlobResponseOp, Boundary,
    Connectivity, Gaussian, GlobalThreshold, GlobalThresholdOutput, SampledValuesOp, SmoothOp,
    ValueSampling, VoxelwiseMapOp, WidenLabelsOp,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::{BlockGrid, Chain, Error, ImageId, PlanBuilder, Result, ZarrEnvironment};
use clap::Parser;
use ndarray::Array3;

use crate::cells::LargeObject;
use crate::measure::{Class, MeasureParams, Measured};
use crate::{beads, cells, measure, ops, qc, style, zio};

/// Label + table names in the dataset.
const BEADS: &str = "bjorn-beads";
const OBJECTS: &str = "bjorn-large-af488";
/// Stream prefix of the surface phases.
const SURFACES: &str = "bjorn.surfaces";

const CH_BEADS: usize = 0;
const CH_EPI: usize = 1;
const CH_MUCUS: usize = 2;

/// Ignore voxels closer than this (µm) above the epithelium when looking for
/// large objects: that is the epithelium itself.
const OBJECT_EPI_MARGIN_UM: f64 = 10.0;
/// ... and look from this many slices above the mucus top.
const OBJECT_ABOVE_MUCUS_SLICES: f32 = 2.0;

/// Everything a run needs; also the command line of the binary.
#[derive(Debug, Clone, Parser)]
#[command(about = "Segment mucus top + epithelium surfaces, detect beads, measure distances")]
pub struct Config {
    /// OME-Zarr image (a local copy: labels and tables are written into it).
    #[arg(long)]
    pub zarr: PathBuf,
    /// Directory for CSV and QC output.
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub workers: usize,
    /// Block edge z,y,x; y and x must be multiples of --stride.
    #[arg(long, value_delimiter = ',', default_values_t = vec![32, 256, 256])]
    pub block: Vec<usize>,
    /// xy pooling between voxels and surface-graph columns.
    #[arg(long, default_value_t = 4)]
    pub stride: usize,
    /// Max z step (slices) between neighbouring graph columns, mucus top.
    #[arg(long, default_value_t = 2)]
    pub delta_mucus: usize,
    /// Max z step (slices) between neighbouring graph columns, epithelium.
    #[arg(long, default_value_t = 2)]
    pub delta_epi: usize,
    /// Weight of the C-Blue (nuclear) cue in the epithelium cost, AF488 = 1.
    #[arg(long, default_value_t = 0.5)]
    pub epi_blue_weight: f64,
    /// Weight of the "AF594 just below" penalty in the epithelium cost.
    #[arg(long, default_value_t = 0.5)]
    pub epi_mucus_below_weight: f64,
    /// Minimum mucus thickness (slices) between mucus top and epithelium.
    #[arg(long, default_value_t = 2)]
    pub min_mucus_slices: usize,
    /// Cost smoothing sigma in voxels, z,y,x.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1.0, 2.0, 2.0])]
    pub cost_sigma: Vec<f64>,
    /// Bead DoG low sigma in voxels, z,y,x (1 µm beads blurred by the PSF).
    #[arg(long, value_delimiter = ',', default_values_t = vec![1.0, 1.2, 1.2])]
    pub bead_sigma: Vec<f64>,
    #[arg(long, default_value_t = 6.0)]
    pub bead_min_distance_um: f64,
    /// Bead threshold: response > median + k * robust sigma (MAD) of the response.
    #[arg(long, default_value_t = 12.0)]
    pub bead_k_mad: f64,
    /// Candidates further than this below the apical surface are in-tissue.
    #[arg(long, default_value_t = 5.0)]
    pub tissue_margin_um: f64,
    /// Large AF488 objects: threshold = median + k * MAD-sigma inside the mucus.
    #[arg(long, default_value_t = 15.0)]
    pub cell_k_mad: f64,
    /// Large AF488 objects: minimum volume (µm³). A blurred bead is ~1000.
    #[arg(long, default_value_t = 4000.0)]
    cell_min_volume_um3: f64,
    /// Fixed min DoG response for beads (default: automatic log-Otsu).
    #[arg(long)]
    pub bead_min_response: Option<f64>,
    /// PSF shape test: min (z neighbours / peak), background subtracted.
    #[arg(long, default_value_t = 0.35)]
    pub bead_min_z_support: f64,
    /// PSF shape test: min (xy neighbours / peak), background subtracted.
    #[arg(long, default_value_t = 0.15)]
    pub bead_min_xy_support: f64,
}

fn triple(v: &[f64], what: &str) -> Result<[f64; 3]> {
    v.try_into()
        .map_err(|_| Error::invalid(format!("{what} needs z,y,x")))
}

/// What the plan produced, read back for CSV output and QC: rows and 2D maps,
/// never volumes.
struct Results {
    heights: measure::HeightMaps,
    beads: Vec<Measured>,
    large: Vec<LargeObject>,
}

/// Counts of what a run found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub candidates: usize,
    pub beads: usize,
    pub in_tissue: usize,
    pub large_objects: usize,
}

/// Run the analysis on `config.zarr`, publish labels and tables into it, and
/// write CSVs and QC images to `config.out`. Replaces any previous output.
pub fn run(config: &Config) -> Result<Summary> {
    let args = config;
    let t0 = Instant::now();
    fs::create_dir_all(&args.out).map_err(Error::backend)?;
    let epi = OmeVolumePyramid::open(&args.zarr, CH_EPI, 0)?;
    let voxel = epi.voxel_size();
    let work = args.out.join(".work");

    let Results {
        heights,
        beads: rows,
        large: object_rows,
    } = run_plan(args, &epi, &work)?;
    eprintln!("plan and publish done {:.1}s", t0.elapsed().as_secs_f64());
    eprintln!(
        "large AF488 objects (dying-cell candidates): {}",
        object_rows.len()
    );
    cells::write_csv(
        &object_rows,
        voxel,
        &args.out.join("large_af488_objects.csv"),
    )?;
    measure::write_csv(&rows, voxel, &args.out.join("beads.csv"))?;
    let n_bead = rows.iter().filter(|m| m.kept()).count();
    let n_tissue = rows.iter().filter(|m| m.class == Class::InTissue).count();
    eprintln!(
        "beads={n_bead} in_tissue={n_tissue} candidates={}",
        rows.len()
    );

    // QC, from windows of the image and the published labels
    let root = epi.image_root();
    let (ny, nx) = heights.epi.dim();
    let nz = epi.levels()[0].shape[0];
    let ys = [ny / 4, ny / 2, 3 * ny / 4];
    let planes = |read: &dyn Fn(i64) -> Result<Array3<f32>>| -> Result<Array3<f32>> {
        let views = ys
            .iter()
            .map(|&y| read(y as i64))
            .collect::<Result<Vec<_>>>()?;
        ndarray::concatenate(
            ndarray::Axis(1),
            &views.iter().map(|v| v.view()).collect::<Vec<_>>(),
        )
        .map_err(Error::backend)
    };
    let channel =
        |c: usize| planes(&|y| zio::read_channel_window(root, c, [0, y, 0], [nz, 1, nx], 0.0));
    let (e, m, b) = (channel(CH_EPI)?, channel(CH_MUCUS)?, channel(CH_BEADS)?);
    let object_planes = ys
        .iter()
        .map(|&y| {
            zio::read_label_window(
                &root.join("labels").join(OBJECTS),
                [0, y as i64, 0],
                [nz, 1, nx],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let objects = ndarray::concatenate(
        ndarray::Axis(1),
        &object_planes.iter().map(|v| v.view()).collect::<Vec<_>>(),
    )
    .map_err(Error::backend)?;
    qc::xz_sections(
        [&e, &m, &b],
        &heights,
        &rows,
        None,
        &ys,
        2,
        &args.out.join("qc_xz.png"),
    )?;
    qc::xz_sections(
        [&e, &m, &b],
        &heights,
        &rows,
        Some(&objects),
        &ys,
        2,
        &args.out.join("qc_xz_objects.png"),
    )?;
    let crops = |sel: &[&Measured]| -> Result<Vec<Array3<f32>>> {
        let h = qc::CROP_HALF;
        sel.iter()
            .map(|r| {
                let start = [
                    r.c.z as i64 - h[0] as i64,
                    r.c.y as i64 - h[1] as i64,
                    r.c.x as i64 - h[2] as i64,
                ];
                zio::read_channel_window(root, CH_BEADS, start, h.map(|v| 2 * v + 1), 50.0)
            })
            .collect()
    };
    let above: Vec<&Measured> = rows
        .iter()
        .filter(|r| r.kept() && r.c.z * voxel[0] < r.h_mucus_um - 50.0)
        .take(24)
        .collect();
    qc::gallery(
        &above,
        &crops(&above)?,
        (50.0, 1500.0),
        &args.out.join("gallery_far_above_mucus.png"),
    )?;
    let mut by_resp: Vec<&Measured> = rows.iter().filter(|r| r.kept()).collect();
    by_resp.sort_by(|a, b| a.c.response.total_cmp(&b.c.response));
    for (lo, hi, name) in [
        (8.0, 16.0, "8_16"),
        (16.0, 32.0, "16_32"),
        (32.0, 64.0, "32_64"),
        (64.0, 128.0, "64_128"),
    ] {
        let sel: Vec<&Measured> = by_resp
            .iter()
            .copied()
            .filter(|r| r.c.response >= lo && r.c.response < hi)
            .step_by(7)
            .take(24)
            .collect();
        qc::gallery(
            &sel,
            &crops(&sel)?,
            (50.0, 1500.0),
            &args.out.join(format!("gallery_resp_{name}.png")),
        )?;
    }
    qc::height_map(&heights.epi, nz, &args.out.join("height_epi.png"))?;
    qc::height_map(&heights.mucus, nz, &args.out.join("height_mucus.png"))?;
    // numeric surfaces (slice index per column), the reference for the rewrite
    blockflow::npy::write_array_file(
        &args.out.join("height_epi.npy"),
        &heights.epi,
        blockflow::npy::Order::C,
    )?;
    blockflow::npy::write_array_file(
        &args.out.join("height_mucus.npy"),
        &heights.mucus,
        blockflow::npy::Order::C,
    )?;
    let _ = fs::remove_dir_all(&work);
    eprintln!("done {:.1}s", t0.elapsed().as_secs_f64());
    Ok(Summary {
        candidates: rows.len(),
        beads: n_bead,
        in_tissue: n_tissue,
        large_objects: object_rows.len(),
    })
}

/// Smooth, log (background ~50 counts; log makes contrasts relative, so depth
/// fading and saturation matter less), then a z filter.
fn cue_chain(sigma: [f64; 3], leaf: Option<Chain>, last: ops::ZTapsOp) -> Result<Chain> {
    let gaussian = Gaussian::new(sigma, 3.0)?.with_boundary(Boundary::Reflect);
    let mut steps = Vec::new();
    steps.extend(leaf);
    steps.push(Chain::op(SmoothOp::new("cue smoothing", gaussian)));
    steps.push(Chain::op(VoxelwiseMapOp::new("log intensity", |v: f64| {
        (v - 30.0).max(1.0).ln()
    })));
    steps.push(Chain::op(last));
    Ok(Chain::sequence(steps))
}

/// The whole analysis as one plan: image 0 = AF488, supplied 0 = AF594,
/// supplied 1 = C-Blue.
fn run_plan(args: &Config, epi: &OmeVolumePyramid, work: &Path) -> Result<Results> {
    let src_epi = epi.attached_level(0)?;
    let src_muc = OmeVolumePyramid::open(&args.zarr, CH_MUCUS, 0)?.attached_level(0)?;
    let src_blue = OmeVolumePyramid::open(&args.zarr, CH_BEADS, 0)?.attached_level(0)?;
    let (dtype, volume) = src_epi.metadata()?;
    let voxel = epi.voxel_size();
    let mucus = ImageId::supplied(0);
    let blue = ImageId::supplied(1);
    let block: [usize; 3] = args
        .block
        .as_slice()
        .try_into()
        .map_err(|_| Error::invalid("--block needs z,y,x"))?;
    let mut b = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, block)?);

    // ---- layers
    let sigma = triple(&args.cost_sigma, "--cost-sigma")?;
    let p_epi = b.pixels(cue_chain(
        sigma,
        None,
        ops::ZTapsOp::edge_cost("epithelium apical edge cost", 1),
    )?)?;
    let p_muc = b.pixels(cue_chain(
        sigma,
        Some(Chain::source(mucus, dtype)),
        ops::ZTapsOp::edge_cost("mucus top edge cost", 1),
    )?)?;
    // tissue under the apical surface is dark in AF594; a mucus lump that also
    // carries AF488 is not
    let p_muc_below = b.pixels(cue_chain(
        sigma,
        Some(Chain::source(mucus, dtype)),
        ops::ZTapsOp::below("mucus stain below", 2),
    )?)?;
    // C-Blue also stains the epithelium (nuclei): a second apical-edge cue that
    // does not bleed into mucus lumps the way AF488 does
    let p_blue = b.pixels(cue_chain(
        sigma,
        Some(Chain::source(blue, dtype)),
        ops::ZTapsOp::edge_cost("epithelium nuclear edge cost", 1),
    )?)?;
    let p_lab = append_layered_surface_phases(
        &mut b,
        SURFACES,
        LayeredSurfaces {
            layers: vec![
                SurfaceLayer {
                    cues: vec![SurfaceCue::raw(p_muc.image()?, 1.0)],
                    weight: 1.0,
                    max_step: args.delta_mucus,
                },
                SurfaceLayer {
                    cues: vec![
                        SurfaceCue::normalised(p_epi.image()?, 1.0),
                        SurfaceCue::normalised(p_blue.image()?, args.epi_blue_weight),
                        SurfaceCue::normalised(p_muc_below.image()?, args.epi_mucus_below_weight),
                    ],
                    weight: 1.0,
                    max_step: args.delta_epi,
                },
            ],
            separations: vec![[args.min_mucus_slices, volume[0]]],
            stride: args.stride,
        },
    )?;

    // published as uint64; a pixel phase reads the image just written
    let p_layers = b.pixels(Chain::op(WidenLabelsOp))?;

    // ---- large AF488 objects inside the mucus band
    let margin = (OBJECT_EPI_MARGIN_UM / voxel[0]) as f32;
    let p_band = b.fragments(SurfaceBandOp::new(
        SURFACES,
        p_lab,
        (0, OBJECT_ABOVE_MUCUS_SLICES),
        (1, margin),
    ))?;
    let smooth = || -> Result<Chain> {
        let gaussian = Gaussian::new([0.5, 1.0, 1.0], 3.0)?.with_boundary(Boundary::Reflect);
        Ok(Chain::op(SmoothOp::new("object smoothing", gaussian)))
    };
    let p_af_smooth = b.pixels(Chain::sequence(vec![
        Chain::source(ImageId::from(0), dtype),
        smooth()?,
    ]))?;
    let p_blue_smooth = b.pixels(Chain::sequence(vec![Chain::source(blue, dtype), smooth()?]))?;
    let in_band = |image: ImageId, name: &'static str, stream: &str| -> Result<SampledValuesOp> {
        Ok(
            SampledValuesOp::new(name, stream, SourceInput::voxelwise(image))
                .on(ValueSampling::CoordinateSum { every: 3 })?
                .within(SourceInput::voxelwise(p_band.image()?)),
        )
    };
    let af_samples = b.fragments(in_band(
        p_af_smooth.image()?,
        "AF488 samples in mucus",
        "bjorn.af488",
    )?)?;
    let blue_samples = b.fragments(in_band(
        p_blue_smooth.image()?,
        "C-Blue samples in mucus",
        "bjorn.cblue",
    )?)?;
    b.fragments(cells::CellMaskOp {
        af488_samples: ("bjorn.af488".into(), af_samples.index()),
        blue_samples: ("bjorn.cblue".into(), blue_samples.index()),
        af488: p_af_smooth.image()?,
        blue: p_blue_smooth.image()?,
        band: p_band.image()?,
        k_mad: args.cell_k_mad,
    })?;
    let faces = b.fragments(
        LabelComponentsOp::new(
            "object components",
            "bjorn.components",
            Lifecycle::DeleteOnExit,
        )
        .connecting(Connectivity::FacesEdgesAndCorners),
    )?;
    let p_components = b.fragments(
        RelabelComponentsOp::reading("object relabel", "bjorn.components", faces, b.grid())
            .connecting(Connectivity::FacesEdgesAndCorners),
    )?;
    let sums = b.fragments(cells::ObjectStatsOp {
        stream: "bjorn.object-sums".into(),
        raw_af488: SourceInput::voxelwise(ImageId::from(0)),
        raw_blue: SourceInput::voxelwise(blue).holding(dtype),
    })?;
    let object_schema = cells::object_schema()?;
    let p_objects = b.fragments(cells::ObjectFilterOp {
        stats: ("bjorn.object-sums".into(), sums.index()),
        components: p_components.image()?,
        min_volume_um3: args.cell_min_volume_um3,
        voxel_volume: voxel.iter().product(),
        rows_stream: "bjorn.objects".into(),
        schema: object_schema.clone(),
    })?;

    // ---- beads
    let bead_sigma = triple(&args.bead_sigma, "--bead-sigma")?;
    let p_dog = b.pixels(Chain::sequence(vec![
        Chain::source(blue, dtype),
        Chain::op(BlobResponseOp::difference_of_gaussians(
            "bead DoG response",
            bead_sigma,
            bead_sigma.map(|s| s * 1.6),
            3.0,
            Boundary::Reflect,
        )?),
    ]))?;
    let every5 = ValueSampling::LinearIndex { every: 5 };
    let response_samples = b.fragments(
        SampledValuesOp::new(
            "bead response samples",
            "bjorn.response",
            SourceInput::voxelwise(p_dog.image()?),
        )
        .on(every5)?,
    )?;
    let background_samples = b.fragments(
        SampledValuesOp::new(
            "C-Blue background samples",
            "bjorn.background",
            SourceInput::voxelwise(blue).holding(dtype),
        )
        .on(every5)?,
    )?;
    // a noise-referenced level (median + k MAD) rather than Otsu, which with
    // saturated beads splits "saturated" from everything else
    b.fragments(
        ApplyGlobalThresholdOp::new(
            "bead response level",
            "bjorn.response",
            response_samples.index(),
            GlobalThreshold::robust_sigma(args.bead_k_mad)?,
            GlobalThresholdOutput::KeepAtOrAbove,
        )
        .reading_image(p_dog.image()?.index()),
    )?;
    let (peaks, peak_schema) = append_response_peak_table_phases(
        &mut b,
        "bjorn.peaks",
        Lifecycle::DeleteOnExit,
        Connectivity::FacesEdgesAndCorners,
        f64::MIN,
    )?;
    let bead_schema = beads::bead_schema()?;
    let select = b.fragments(beads::BeadSelectOp {
        peaks_stream: "bjorn.peaks".into(),
        peaks_phase: peaks.index(),
        peaks_schema: peak_schema,
        background_stream: "bjorn.background".into(),
        background_phase: background_samples.index(),
        heights_stream: format!("{SURFACES}.heights"),
        heights_phase: p_lab.index(),
        raw: blue,
        raw_dtype: dtype,
        rows_stream: "bjorn.selected".into(),
        params: beads::BeadSelectParams {
            min_distance_um: args.bead_min_distance_um,
            measure: MeasureParams {
                voxel,
                tissue_margin_um: args.tissue_margin_um,
                min_z_support: args.bead_min_z_support,
                min_xy_support: args.bead_min_xy_support,
            },
        },
        schema: bead_schema.clone(),
    })?;
    let fin = b.fragments(beads::BeadFinalOp {
        select_stream: "bjorn.selected".into(),
        select_phase: select.index(),
        objects: p_objects.image()?,
        rows_stream: "bjorn.beads".into(),
        min_response: args.bead_min_response,
        schema: bead_schema.clone(),
    })?;

    // ---- what is published
    let p_lines = b.fragments(SurfaceLinesOp::new(SURFACES, p_lab))?;
    let p_bead_labels = b.fragments(beads::BeadLabelsOp {
        rows: ("bjorn.beads".into(), fin.index()),
        schema: bead_schema.clone(),
        radius: [1.0, 2.0, 2.0],
    })?;
    let final_object_schema = cells::final_object_schema()?;
    let counts = b.fragments(cells::ObjectCountsOp {
        objects: ("bjorn.objects".into(), p_objects.index()),
        beads: ("bjorn.beads".into(), fin.index()),
        bead_schema: bead_schema.clone(),
        bead_object_column: 6,
        object_schema: object_schema.clone(),
        rows_stream: "bjorn.objects.final".into(),
        schema: final_object_schema.clone(),
    })?;

    // ---- run
    let assembly = b.finish()?;
    if work.exists() {
        fs::remove_dir_all(work).map_err(Error::backend)?;
    }
    let env = ZarrEnvironment::attach(work, &[src_epi, src_muc, src_blue])?;
    let published = [
        (&p_layers, "bjorn-layers"),
        (&p_lines, "bjorn-surfaces"),
        (&p_objects, OBJECTS),
        (&p_bead_labels, BEADS),
    ];
    let hints = Hints {
        concurrency: args.workers,
        keep_images: published
            .iter()
            .map(|(p, _)| p.image())
            .collect::<Result<_>>()?,
        ..Hints::default()
    };
    execute_phases(
        "bjorn",
        &assembly.workflow,
        &assembly.decomposition,
        &hints,
        &env,
        &[],
        &assembly.work(),
    )?;

    // read back rows and the solved surfaces
    let mut surfaces = None;
    fold_fragments(&env, &format!("{SURFACES}.heights"), &mut |key, bytes| {
        if key.phase == p_lab.index() {
            surfaces = Some(SurfaceColumns::decode(bytes)?);
        }
        Ok(())
    })?;
    let surfaces =
        surfaces.ok_or_else(|| Error::invalid("the surface solve published no heights"))?;
    let mut bead_rows = collect_rows(
        &env,
        "bjorn.beads",
        fin.index(),
        volume,
        bead_schema.clone(),
    )?
    .into_iter()
    .map(|r| beads::measured_of(r.at, &r.values))
    .collect::<Result<Vec<_>>>()?;
    // strongest first, as the selection walked them
    bead_rows.sort_by(|a, b| {
        b.c.response.total_cmp(&a.c.response).then(
            (a.c.z, a.c.y, a.c.x)
                .partial_cmp(&(b.c.z, b.c.y, b.c.x))
                .unwrap(),
        )
    });
    let mut large = collect_rows(
        &env,
        "bjorn.objects.final",
        counts.index(),
        volume,
        final_object_schema.clone(),
    )?
    .iter()
    .map(|r| cells::object_of(&r.values, voxel))
    .collect::<Result<Vec<_>>>()?;
    large.sort_by_key(|o| o.id);

    // tables, streamed from the row fragments
    let tables = work.join("tables");
    let mut bead_spec = style::bead_table_spec(volume);
    bead_spec.region = Some(format!("../../labels/{BEADS}"));
    finalize_object_table(
        &env,
        "bjorn.beads",
        fin.index(),
        volume,
        bead_schema,
        tables.join(BEADS),
        work,
        bead_spec,
        |row| Ok(Some(style::bead_values(&beads::measured_from_row(row)?))),
    )?;
    let mut object_spec = style::object_table_spec(volume);
    object_spec.region = Some(format!("../../labels/{OBJECTS}"));
    finalize_object_table(
        &env,
        "bjorn.objects.final",
        counts.index(),
        volume,
        final_object_schema,
        tables.join(OBJECTS),
        work,
        object_spec,
        |row| {
            Ok(Some(style::object_values(&cells::object_of(
                &beads::values_of_row(row)?,
                voxel,
            )?)))
        },
    )?;
    drop(env);

    // install the label images (moved from the work directory) and tables
    for (phase, name) in published {
        let table = tables.join(name);
        let style = match name {
            "bjorn-layers" => style::layer_style().as_object().cloned(),
            "bjorn-surfaces" => style::surface_style().as_object().cloned(),
            _ => None,
        };
        publish_label(
            epi,
            LabelPublication {
                name,
                level0: &work.join(format!("level{}", phase.image()?.index())),
                image_label: style,
                table: table.exists().then_some(table.as_path()),
            },
            args.workers,
        )?;
    }
    Ok(Results {
        heights: measure::heights_from_columns(&surfaces, volume[1], volume[2]),
        beads: bead_rows,
        large,
    })
}
