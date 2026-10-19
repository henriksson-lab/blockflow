//! Find round, thick-walled shells (cysts) in a 3D OME-Zarr image without
//! learning, planned and out of core, and publish them as an editable label
//! layer with its object table. See `SHELL_OBJECTS_3D_PLAN.md`.
//!
//! 1. block-mean the image to an isotropic coarse grid (one run, kept);
//! 2. scan it with a radial shell template: NCC at several radii, peaks,
//!    suppression (one run per template);
//! 3. at level 0: smooth, fit every candidate along rays, filter, drop
//!    duplicates, paint labels (one run per pass);
//! 4. optionally rebuild the template from the best fits and repeat 2–3.
//!
//! All lengths are level-0 voxels. The voxel grid is assumed isotropic in
//! space; the stored spacing is used only for the µm columns of the table, and
//! `--voxel-size` overrides it.

mod sheet;

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use blockflow::fragment::fold_fragments;
use blockflow::object_table::{
    finalize_object_table, ColumnRole, ColumnSpec, CoordinateColumn, DType, ObjectValue,
    SpatialIndexSpec, TableSpec,
};
use blockflow::ome_zarr::{publish_label, LabelPublication, OmeVolumePyramid};
use blockflow::ops::rows::{collect_rows, RowFilter, RowSourceOp, RowStreams, RowValues};
use blockflow::ops::{
    append_block_mean_phases, append_response_peak_table_phases, decode_shapes, fit_shell,
    gaussian_smooth_into_with, multi_radius_ncc_chain, radial_profile, shape_label, Boundary,
    ColumnTest, Connectivity, FilterRowsOp, Gaussian, RadialTemplate, RadiusPair, ShellFit,
    ShellFitOp, ShellFitParams, ShellJudge, StarConvexLabelsOp, StarShape, SuppressRowsOp,
    Suppression, SuppressionRadius,
};
use blockflow::sidecar::Lifecycle;
use blockflow::sphere::{fibonacci_sphere, RaySet};
use blockflow::strategy::{execute_phases, Hints};
use blockflow::table::{Column, Schema, Value};
use blockflow::{AttachedImage, BlockGrid, Error, ImageId, PlanBuilder, Result, ZarrEnvironment};
use clap::Parser;

const PEAKS: &str = "scan.peaks";
const SCAN: &str = "scan.candidates";
const CANDIDATES: &str = "shell.candidates";
const FITTED: &str = "shell.fitted";
const SHAPES: &str = "shell.shapes";
const ACCEPTED: &str = "shell.accepted";
const KEPT: &str = "shell.kept";
/// Template sample spacing and extent, in r / R.
const TEMPLATE_STEP: f64 = 0.02;
const TEMPLATE_EXTENT: f64 = 1.6;

#[derive(Debug, Parser)]
#[command(about = "Detect and outline round shells (cysts) in a 3D OME-Zarr image")]
struct Args {
    /// OME-Zarr image (the multiscale group or its bioformats2raw parent).
    #[arg(long)]
    zarr: PathBuf,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    #[arg(long, default_value_t = 0)]
    time: usize,
    /// Level-0 voxel centre z,y,x of one example object, to build the template.
    #[arg(long, value_delimiter = ',')]
    exemplar: Option<Vec<f64>>,
    /// A template saved by an earlier run, instead of --exemplar.
    #[arg(long)]
    template: Option<PathBuf>,
    /// Expected outer diameter in level-0 voxels (for the exemplar fit).
    #[arg(long, default_value_t = 56.0)]
    diameter: f64,
    /// Isotropic downsampling of the scan, in level-0 voxels.
    #[arg(long, default_value_t = 4)]
    scan_factor: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0.75,0.85,0.95,1.05,1.15,1.3"
    )]
    scales: Vec<f64>,
    /// Template support in r / R; the interior below the first value is ignored.
    #[arg(long, value_delimiter = ',', default_value = "0.65,1.4")]
    band: Vec<f64>,
    /// Lowest scan score kept as a candidate. Low on purpose: the fit decides.
    #[arg(long, default_value_t = 0.10)]
    min_score: f64,
    #[arg(long, default_value_t = 600)]
    max_candidates: usize,
    /// Rebuild the template from this many of the best fits and rescan (0: off).
    #[arg(long, default_value_t = 12)]
    bootstrap: usize,
    #[arg(long, default_value_t = 500)]
    rays: usize,
    #[arg(long, default_value_t = 0.5)]
    min_coverage: f64,
    #[arg(long, default_value_t = 0.2)]
    max_roughness: f64,
    #[arg(long, default_value_t = 0.03)]
    min_contrast: f64,
    #[arg(long, default_value_t = 0.4)]
    strong_step: f64,
    /// Lowest fraction of strong wall rays. Tuned for recall on ND_WT_5_s1
    /// (2026-10-19): 0.45/0.3 kept 44, these keep 54, of which about 5 more
    /// real cysts (mostly small ones) and 5 more false positives.
    #[arg(long, default_value_t = 0.35)]
    min_strong: f64,
    /// The same for objects of at least three quarters of the exemplar's size.
    #[arg(long, default_value_t = 0.28)]
    min_strong_large: f64,
    /// Ring depth relative to the exemplar's wall step.
    #[arg(long, default_value_t = 0.15)]
    min_ring_depth: f64,
    #[arg(long, default_value_t = 1.0)]
    wall_pad: f64,
    /// Table column that decides which of two overlapping accepted fits is
    /// kept. `strong_margin` (wall evidence above the fit's own size-dependent
    /// threshold) prefers the whole cyst over an undersized fit inside it,
    /// which the scan `score` can rank higher.
    #[arg(long, default_value = "strong_margin")]
    duplicate_priority: String,
    /// Level-0 block edge z,y,x of the fit plan. Default: four times the fit's
    /// reach, rounded up to a multiple of 32, which keeps the window a block
    /// reads within a few times its core.
    #[arg(long, value_delimiter = ',')]
    block: Option<Vec<usize>>,
    /// Block edge of the coarse plans. A whole number of scan tiles, so the
    /// scan's halo is the kernel's reach and not a tile more.
    #[arg(long, default_value_t = 128)]
    scan_block: usize,
    /// Edge of the scan's transform tiles (coarse voxels).
    #[arg(long, default_value_t = 64)]
    scan_tile: usize,
    /// Physical voxel size z,y,x for the table's µm columns; default the stored one.
    #[arg(long, value_delimiter = ',')]
    voxel_size: Option<Vec<f64>>,
    /// Label and table layer name; an existing layer of this name is replaced.
    #[arg(long, default_value = "shell-objects")]
    layer: String,
    #[arg(long)]
    no_publish: bool,
    /// Also write the per-ray radii of the published objects to the output directory.
    #[arg(long)]
    keep_rays: bool,
    /// Local directory for the CSV, template and contact sheets.
    #[arg(long, default_value = "results")]
    out: PathBuf,
    #[arg(long, default_value_t = 16)]
    workers: usize,
    /// Viewer asked to rescan after publishing (host:port); empty disables.
    #[arg(long, default_value = "beagle.henlab.org:9876")]
    viewer: String,
}

struct Candidate {
    at: [usize; 3],
    score: f64,
}

/// One fitted candidate after a level-0 pass.
struct Object {
    at: [usize; 3],
    values: Vec<Value>,
    shape: StarShape,
    accepted: bool,
    kept: bool,
}

impl Object {
    fn f(&self, schema: &Schema, column: &str) -> f64 {
        match self.values[schema.index_of(column).expect("a fit column")] {
            Value::F64(v) => v,
            Value::U64(v) => v as f64,
        }
    }
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("build with --release"));
    }
    let args = Args::parse();
    validate(&args)?;
    let started = Instant::now();
    let pyramid = OmeVolumePyramid::open(&args.zarr, args.channel, args.time)?;
    let volume = pyramid.levels()[0].shape;
    let voxel_size: [f64; 3] = match &args.voxel_size {
        Some(v) => [v[0], v[1], v[2]],
        None => pyramid.voxel_size(),
    };
    let work = pyramid
        .image_root()
        .join("tables")
        .join(format!(".{}-blockflow-work", args.layer));
    if work.exists() {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }
    fs::create_dir_all(&args.out).map_err(Error::backend)?;
    let name = args
        .zarr
        .file_name()
        .map(|s| s.to_string_lossy().trim_end_matches(".ome.zarr").to_owned())
        .unwrap_or_default();
    let stem = args.out.join(format!("{name}__{}", args.layer));
    let f = args.scan_factor as f64;

    // 1. the coarse grid
    let coarse = block_mean(&pyramid, args.scan_factor, &work.join("mean"), &args)?;
    let (_, coarse_volume) = coarse.metadata()?;
    let variance = sample_variance(&coarse, coarse_volume)?;
    eprintln!(
        "coarse {coarse_volume:?} at {}x, variance {variance:.1}, {:.1}s",
        args.scan_factor,
        started.elapsed().as_secs_f64()
    );

    // the template and the reference radius and wall step
    let profile_rays = fibonacci_sphere(500);
    let profile_at = |centre: [f64; 3], radius: f64| -> Result<RadialTemplate> {
        coarse_profile(&coarse, coarse_volume, centre, radius, f, &profile_rays)
    };
    let (mut template, radius, step_reference) = match (&args.exemplar, &args.template) {
        (_, Some(path)) => {
            let j: serde_json::Value =
                serde_json::from_slice(&fs::read(path).map_err(Error::backend)?)
                    .map_err(Error::backend)?;
            let values = j["values"]
                .as_array()
                .ok_or_else(|| Error::invalid("template has no values"))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            let field = |k: &str| {
                j[k].as_f64()
                    .ok_or_else(|| Error::invalid(format!("template has no {k}")))
            };
            (
                RadialTemplate::new(field("step")?, values)?,
                field("radius")?,
                field("step_ref")?,
            )
        }
        (Some(e), None) => {
            let fit = host_fit(
                &pyramid,
                [e[0], e[1], e[2]],
                &ShellFitParams {
                    wall_pad: args.wall_pad,
                    ..ShellFitParams::for_radius(args.diameter / 2.0)
                },
            )?;
            eprintln!(
                "exemplar: centre {:.1?} mean r {:.1} (std {:.2}) coverage {:.2} wall step {:.0}",
                fit.center, fit.mean_r, fit.std_r, fit.coverage, fit.step_median
            );
            (
                profile_at(fit.center, fit.mean_r)?,
                fit.mean_r,
                fit.step_median,
            )
        }
        (None, None) => return Err(Error::invalid("give --exemplar or --template")),
    };
    let params = ShellFitParams {
        wall_pad: args.wall_pad,
        ..ShellFitParams::for_radius(radius)
    };
    let judge = ShellJudge {
        step_reference,
        strong_step: args.strong_step,
        min_strong: args.min_strong,
        min_strong_large: args.min_strong_large,
        large_from: 0.75,
    };

    // 2-3, and the bootstrap
    let mut candidates = scan(
        &coarse,
        coarse_volume,
        &template,
        radius,
        variance,
        volume,
        &work.join("scan"),
        &args,
    )?;
    // without a bootstrap the first pass is the last, and it paints
    let single = args.bootstrap == 0;
    let mut pass = fit_pass(
        &pyramid,
        &candidates,
        &params,
        &judge,
        radius,
        &work.join("fit"),
        &args,
        single && !args.no_publish,
    )?;
    if !single {
        let mut best: Vec<&Object> = pass.objects.iter().filter(|o| o.kept).collect();
        best.sort_by(|a, b| {
            b.f(&pass.schema, "coverage")
                .total_cmp(&a.f(&pass.schema, "coverage"))
                .then(a.at.cmp(&b.at))
        });
        best.truncate(args.bootstrap);
        if best.len() >= 3 {
            let profiles = best
                .iter()
                .map(|o| profile_at(o.shape.center, o.f(&pass.schema, "mean_r")))
                .collect::<Result<Vec<_>>>()?;
            template = RadialTemplate::average(&profiles, [0.0, TEMPLATE_EXTENT])?;
            let first = pass.objects.iter().filter(|o| o.kept).count();
            eprintln!(
                "bootstrap: template from {} objects (first pass kept {first}); rescanning",
                best.len()
            );
            candidates = scan(
                &coarse,
                coarse_volume,
                &template,
                radius,
                variance,
                volume,
                &work.join("scan"),
                &args,
            )?;
        }
        pass = fit_pass(
            &pyramid,
            &candidates,
            &params,
            &judge,
            radius,
            &work.join("fit"),
            &args,
            !args.no_publish,
        )?;
    }
    let kept: Vec<&Object> = pass.objects.iter().filter(|o| o.kept).collect();
    eprintln!(
        "kept {} of {} candidates ({} accepted before duplicates)",
        kept.len(),
        pass.objects.len(),
        pass.objects.iter().filter(|o| o.accepted).count()
    );

    // reports
    let mut tj = serde_json::json!({"step": template.step, "values": template.values});
    tj["radius"] = radius.into();
    tj["step_ref"] = step_reference.into();
    fs::write(
        stem.with_extension("template.json"),
        serde_json::to_vec_pretty(&tj).expect("json"),
    )
    .map_err(Error::backend)?;
    let t = Instant::now();
    write_csv(&stem.with_extension("csv"), &pass, &args, radius, volume)?;
    let rays = RaySet::fibonacci(args.rays, 8);
    let half = (1.5 * radius).ceil() as usize;
    fn tiles<'a>(objects: &[&'a Object]) -> Vec<sheet::Tile<'a>> {
        objects
            .iter()
            .map(|o| sheet::Tile {
                center: o.shape.center,
                radii: &o.shape.radii,
                accepted: o.kept,
            })
            .collect()
    }
    sheet::contact_sheet(
        &pyramid,
        &stem.with_extension("accepted.png"),
        &tiles(&kept),
        &rays,
        half,
    )?;
    let mut near: Vec<&Object> = pass.objects.iter().filter(|o| !o.accepted).collect();
    near.sort_by(|a, b| {
        b.f(&pass.schema, "strong")
            .total_cmp(&a.f(&pass.schema, "strong"))
            .then(a.at.cmp(&b.at))
    });
    near.truncate(48);
    sheet::contact_sheet(
        &pyramid,
        &stem.with_extension("rejected.png"),
        &tiles(&near),
        &rays,
        half,
    )?;
    if args.keep_rays {
        write_rays(&stem.with_extension("rays.csv"), &kept, &rays, volume)?;
    }

    eprintln!("reports in {:.1}s", t.elapsed().as_secs_f64());
    if let Some((env, label_image, kept_phase)) = pass.publish {
        let t = Instant::now();
        publish(
            &pyramid,
            &env,
            &work,
            label_image,
            kept_phase,
            &pass.schema,
            volume,
            voxel_size,
            &args,
        )?;
        eprintln!("published in {:.1}s", t.elapsed().as_secs_f64());
        if !args.viewer.is_empty() {
            rescan_viewer(&args.viewer);
        }
    }
    fs::remove_dir_all(&work).map_err(Error::backend)?;
    eprintln!("done in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn validate(args: &Args) -> Result<()> {
    let ok = args.scan_factor > 0
        && !args.scales.is_empty()
        && args.band.len() == 2
        && args
            .block
            .as_ref()
            .is_none_or(|b| b.len() == 3 && b.iter().all(|&v| v > 0))
        && args.rays >= 20
        && args.exemplar.as_ref().is_none_or(|e| e.len() == 3)
        && args.voxel_size.as_ref().is_none_or(|v| v.len() == 3)
        && !args.layer.is_empty()
        && !args.layer.contains('/');
    if ok {
        Ok(())
    } else {
        Err(Error::invalid("invalid arguments; see --help"))
    }
}

/// The coarsest stored level whose factors divide `factor`, block-averaged for
/// the remainder in a planned run; the result is kept and attached.
fn block_mean(
    pyramid: &OmeVolumePyramid,
    factor: usize,
    work: &Path,
    args: &Args,
) -> Result<AttachedImage> {
    let s0 = pyramid.levels()[0].shape;
    let (level, stored) = pyramid
        .levels()
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (
                i,
                std::array::from_fn::<usize, 3, _>(|a| {
                    (s0[a] as f64 / l.shape[a] as f64).round() as usize
                }),
            )
        })
        .filter(|(_, f)| f.iter().all(|&v| v > 0 && factor.is_multiple_of(v)))
        .max_by_key(|(_, f)| f[0] * f[1] * f[2])
        .ok_or_else(|| Error::invalid("no stored level divides the scan factor"))?;
    let remainder: [usize; 3] = std::array::from_fn(|a| factor / stored[a]);
    eprintln!("scan: level {level} (factors {stored:?}), block mean {remainder:?}");
    let source = pyramid.attached_level(level)?;
    let (dtype, input) = source.metadata()?;
    let block = args.scan_block.max(1);
    let mut builder = PlanBuilder::new(
        input,
        dtype,
        BlockGrid::new(
            input,
            [
                block.min(input[0]),
                block.min(input[1]),
                block.min(input[2]),
            ],
        )?,
    );
    let mean = append_block_mean_phases(&mut builder, remainder, [block; 3])?;
    let assembly = builder.finish()?;
    let env = ZarrEnvironment::attach(work, std::slice::from_ref(&source))?;
    execute_phases(
        "shell objects: block mean",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: args.workers.max(1),
            keep_images: [mean.image()?].into_iter().collect(),
            ..Hints::default()
        },
        &env,
        &[],
        &assembly.work(),
    )?;
    Ok(AttachedImage::at(
        work.join(format!("level{}", mean.image()?.index())),
    ))
}

/// Variance of the coarse image, from a central box of at most 128^3 voxels:
/// the NCC floor is a fraction of it.
fn sample_variance(coarse: &AttachedImage, volume: [usize; 3]) -> Result<f64> {
    let shape: [usize; 3] = std::array::from_fn(|a| volume[a].min(128));
    let start: [usize; 3] = std::array::from_fn(|a| (volume[a] - shape[a]) / 2);
    let v = coarse.read_region(start, shape)?.widened();
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    Ok(v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n)
}

/// Radial profile on the coarse grid around a level-0 centre and radius.
fn coarse_profile(
    coarse: &AttachedImage,
    volume: [usize; 3],
    centre: [f64; 3],
    radius: f64,
    f: f64,
    directions: &[[f64; 3]],
) -> Result<RadialTemplate> {
    let c: [f64; 3] = std::array::from_fn(|a| (centre[a] - (f - 1.0) / 2.0) / f);
    let reach = (radius / f * TEMPLATE_EXTENT).ceil() as isize + 2;
    let start: [usize; 3] = std::array::from_fn(|a| {
        (c[a].round() as isize - reach).clamp(0, volume[a] as isize - 1) as usize
    });
    let end: [usize; 3] = std::array::from_fn(|a| {
        ((c[a].round() as isize + reach + 1).max(1) as usize).min(volume[a])
    });
    let shape: [usize; 3] = std::array::from_fn(|a| (end[a] - start[a]).max(1));
    let window = coarse.read_region(start, shape)?.widened();
    let local: [f64; 3] = std::array::from_fn(|a| c[a] - start[a] as f64);
    radial_profile(
        window.view(),
        local,
        radius / f,
        directions,
        TEMPLATE_STEP,
        TEMPLATE_EXTENT,
    )
}

/// The exemplar's fit, host-side on a smoothed level-0 window.
fn host_fit(
    pyramid: &OmeVolumePyramid,
    centre: [f64; 3],
    params: &ShellFitParams,
) -> Result<ShellFit> {
    let volume = pyramid.levels()[0].shape;
    let half = params.reach() as isize + 4;
    let start: [usize; 3] = std::array::from_fn(|a| {
        (centre[a].round() as isize - half).clamp(0, volume[a] as isize - 1) as usize
    });
    let end: [usize; 3] = std::array::from_fn(|a| {
        ((centre[a].round() as isize + half + 1).max(1) as usize).min(volume[a])
    });
    let shape: [usize; 3] = std::array::from_fn(|a| end[a] - start[a]);
    let raw = pyramid
        .attached_level(0)?
        .read_region(start, shape)?
        .widened();
    let mut smoothed = ndarray::Array3::<f64>::zeros(raw.raw_dim());
    gaussian_smooth_into_with(
        raw.view(),
        Gaussian::isotropic(1.0, 4.0)?.kernels(),
        Boundary::Clamp,
        smoothed.view_mut(),
    )?;
    fit_shell(
        smoothed.view(),
        start,
        volume,
        centre,
        &RaySet::fibonacci(500, 8),
        &RaySet::fibonacci(120, 8),
        params,
    )
}

/// One template scan of the coarse image: candidates in level-0 voxels,
/// strongest first.
#[allow(clippy::too_many_arguments)]
fn scan(
    coarse: &AttachedImage,
    coarse_volume: [usize; 3],
    template: &RadialTemplate,
    radius: f64,
    variance: f64,
    volume: [usize; 3],
    work: &Path,
    args: &Args,
) -> Result<Vec<Candidate>> {
    let t = Instant::now();
    if work.exists() {
        fs::remove_dir_all(work).map_err(Error::backend)?;
    }
    let f = args.scan_factor as f64;
    let radii: Vec<f64> = args.scales.iter().map(|s| s * radius / f).collect();
    let (dtype, _) = coarse.metadata()?;
    let block: [usize; 3] = std::array::from_fn(|a| args.scan_block.min(coarse_volume[a]));
    let mut builder = PlanBuilder::new(coarse_volume, dtype, BlockGrid::new(coarse_volume, block)?);
    builder.pixels(multi_radius_ncc_chain(
        template,
        &radii,
        [args.band[0], args.band[1]],
        [args.scan_tile.max(1); 3],
        0.01 * variance,
    )?)?;
    let (peaks, schema) = append_response_peak_table_phases(
        &mut builder,
        PEAKS,
        Lifecycle::DeleteOnExit,
        Connectivity::FacesEdgesAndCorners,
        args.min_score,
    )?;
    let smallest = radii.iter().copied().fold(f64::INFINITY, f64::min);
    let suppressed = builder.fragments(SuppressRowsOp::new(
        "scan suppression",
        RowStreams::new(
            PEAKS,
            peaks.index(),
            SCAN,
            Lifecycle::Persistent,
            schema.clone(),
        )?,
        Suppression {
            priority: "response".into(),
            radius: SuppressionRadius::Fixed(0.75 * smallest),
            pair: RadiusPair::Smaller,
            factor: 1.0,
            voxel_size: [1.0; 3],
            position: None,
        },
    )?)?;
    let assembly = builder.finish()?;
    let env = ZarrEnvironment::attach(work, std::slice::from_ref(coarse))?;
    execute_phases(
        "shell objects: template scan",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: args.workers.max(1),
            ..Hints::default()
        },
        &env,
        &[],
        &assembly.work(),
    )?;
    let response = schema
        .index_of("response")
        .expect("peak rows carry a response");
    let mut found: Vec<Candidate> =
        collect_rows(&env, SCAN, suppressed.index(), coarse_volume, schema)?
            .into_iter()
            .map(|row| Candidate {
                at: std::array::from_fn(|a| {
                    ((row.at[a] as f64 * f + (f - 1.0) / 2.0).round() as usize).min(volume[a] - 1)
                }),
                score: match row.values[response] {
                    Value::F64(v) => v,
                    Value::U64(v) => v as f64,
                },
            })
            .collect();
    found.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.at.cmp(&b.at)));
    found.truncate(args.max_candidates);
    eprintln!(
        "scan: {} candidates in {:.1}s",
        found.len(),
        t.elapsed().as_secs_f64()
    );
    Ok(found)
}

struct Pass {
    schema: Schema,
    objects: Vec<Object>,
    /// The run's environment, the label image and the phase of the kept rows,
    /// when the pass painted.
    publish: Option<(ZarrEnvironment, ImageId, usize)>,
}

/// One level-0 pass: smooth, fit, accept, de-duplicate and, if `paint`, label.
#[allow(clippy::too_many_arguments)]
fn fit_pass(
    pyramid: &OmeVolumePyramid,
    candidates: &[Candidate],
    params: &ShellFitParams,
    judge: &ShellJudge,
    radius: f64,
    work: &Path,
    args: &Args,
    paint: bool,
) -> Result<Pass> {
    let t = Instant::now();
    if work.exists() {
        fs::remove_dir_all(work).map_err(Error::backend)?;
    }
    let source = pyramid.attached_level(0)?;
    let (dtype, volume) = source.metadata()?;
    let edge = (4 * params.reach()).div_ceil(32) * 32;
    let block: [usize; 3] =
        std::array::from_fn(|a| args.block.as_ref().map_or(edge, |b| b[a]).min(volume[a]));
    let mut builder = PlanBuilder::new(volume, dtype, BlockGrid::new(volume, block)?);
    let candidate_schema = Schema::new(vec![Column::f64("score")])?;
    let rows: Vec<RowValues> = candidates
        .iter()
        .map(|c| RowValues::new(c.at, vec![Value::F64(c.score)]))
        .collect();
    let sourced = builder.fragments(RowSourceOp::new(
        "shell candidates",
        CANDIDATES,
        Lifecycle::DeleteOnExit,
        candidate_schema.clone(),
        rows,
    ))?;
    let fit = ShellFitOp::new(
        "shell fit",
        ImageId::from(0),
        (CANDIDATES, sourced.index()),
        candidate_schema,
        (FITTED, Lifecycle::Persistent),
        (SHAPES, Lifecycle::Persistent),
        params.clone(),
        *judge,
        args.rays,
    )?
    .with_prefilter(Gaussian::isotropic(1.0, 4.0)?);
    let schema = fit.schema().clone();
    let rays: Arc<RaySet> = fit.rays().clone();
    let fitted = builder.fragments(fit)?;
    let filter = RowFilter::new(vec![
        ColumnTest::at_least("strong_margin", 0.0)?,
        ColumnTest::at_least("ring_depth", args.min_ring_depth)?,
        ColumnTest::at_least("coverage", args.min_coverage)?,
        ColumnTest::range("roughness", 0.0, args.max_roughness)?,
        ColumnTest::range("mean_r", 0.6 * radius, 1.6 * radius)?,
        ColumnTest::at_least("contrast", args.min_contrast)?,
    ])?;
    let accepted = builder.fragments(FilterRowsOp::new(
        "shell acceptance",
        RowStreams::new(
            FITTED,
            fitted.index(),
            ACCEPTED,
            Lifecycle::Persistent,
            schema.clone(),
        )?,
        filter,
    )?)?;
    // an object centred inside a better-scoring one is part of it
    let kept = builder.fragments(SuppressRowsOp::new(
        "shell duplicates",
        RowStreams::new(
            ACCEPTED,
            accepted.index(),
            KEPT,
            Lifecycle::Persistent,
            schema.clone(),
        )?,
        Suppression {
            priority: args.duplicate_priority.clone(),
            radius: SuppressionRadius::Column("mean_r".into()),
            pair: RadiusPair::Larger,
            factor: 0.9,
            voxel_size: [1.0; 3],
            position: Some(["center_z".into(), "center_y".into(), "center_x".into()]),
        },
    )?)?;
    let labels = if paint {
        Some(builder.fragments(StarConvexLabelsOp::new(
            "shell labels",
            (KEPT, kept.index()),
            schema.clone(),
            (SHAPES, fitted.index()),
            rays,
        ))?)
    } else {
        None
    };
    let assembly = builder.finish()?;
    let env = ZarrEnvironment::attach(work, std::slice::from_ref(&source))?;
    let mut keep = BTreeSet::new();
    if let Some(labels) = &labels {
        keep.insert(labels.image()?);
    }
    execute_phases(
        "shell objects: level-0 fit",
        &assembly.workflow,
        &assembly.decomposition,
        &Hints {
            concurrency: args.workers.max(1),
            keep_images: keep,
            ..Hints::default()
        },
        &env,
        &[],
        &assembly.work(),
    )?;
    let set = |stream: &str, phase: usize| -> Result<BTreeSet<[usize; 3]>> {
        Ok(collect_rows(&env, stream, phase, volume, schema.clone())?
            .into_iter()
            .map(|r| r.at)
            .collect())
    };
    let accepted_set = set(ACCEPTED, accepted.index())?;
    let kept_set = set(KEPT, kept.index())?;
    let mut shapes = std::collections::BTreeMap::new();
    fold_fragments(&env, SHAPES, &mut |key, bytes| {
        if key.phase == fitted.index() {
            for s in decode_shapes(bytes)?.1 {
                shapes.insert(s.at, s);
            }
        }
        Ok(())
    })?;
    let mut objects: Vec<Object> =
        collect_rows(&env, FITTED, fitted.index(), volume, schema.clone())?
            .into_iter()
            .map(|row| {
                let shape = shapes
                    .remove(&row.at)
                    .ok_or_else(|| Error::invalid(format!("no shape for {:?}", row.at)))?;
                Ok(Object {
                    at: row.at,
                    accepted: accepted_set.contains(&row.at),
                    kept: kept_set.contains(&row.at),
                    values: row.values,
                    shape,
                })
            })
            .collect::<Result<_>>()?;
    let score = schema.index_of("score").expect("score column");
    objects.sort_by(|a, b| {
        let s = |o: &Object| match o.values[score] {
            Value::F64(v) => v,
            Value::U64(v) => v as f64,
        };
        s(b).total_cmp(&s(a)).then(a.at.cmp(&b.at))
    });
    eprintln!(
        "fit: {} candidates, {} accepted, {} kept, {:.1}s",
        objects.len(),
        accepted_set.len(),
        kept_set.len(),
        t.elapsed().as_secs_f64()
    );
    let publish = match labels {
        Some(labels) => Some((env, labels.image()?, kept.index())),
        None => None,
    };
    Ok(Pass {
        schema,
        objects,
        publish,
    })
}

/// The kept rows as `tables/<layer>` and the label image as `labels/<layer>`.
#[allow(clippy::too_many_arguments)]
fn publish(
    pyramid: &OmeVolumePyramid,
    env: &ZarrEnvironment,
    work: &Path,
    label_image: ImageId,
    kept_phase: usize,
    schema: &Schema,
    volume: [usize; 3],
    voxel_size: [f64; 3],
    args: &Args,
) -> Result<()> {
    // the fitted centre, which may lie outside the volume for an object cut by a
    // face; `centroid_*` is that centre clamped into the volume, as the spatial
    // index requires
    let measurements = [
        "center_z",
        "center_y",
        "center_x",
        "score",
        "mean_r",
        "std_r",
        "min_r",
        "max_r",
        "roughness",
        "coverage",
        "strong",
        "strong_margin",
        "ring_depth",
        "contrast",
        "shift",
    ];
    let mut columns = vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_z", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("diameter_um", DType::F32, ColumnRole::Measurement),
    ];
    columns.extend(
        measurements
            .iter()
            .map(|m| ColumnSpec::new(*m, DType::F32, ColumnRole::Measurement)),
    );
    let tile_shape = vec![32, 256, 256];
    let spatial_index = SpatialIndexSpec {
        coordinates: ["z", "y", "x"]
            .into_iter()
            .map(|axis| CoordinateColumn {
                axis: axis.into(),
                column: format!("centroid_{axis}"),
            })
            .collect(),
        grid_shape: (0..3)
            .map(|a| volume[a].div_ceil(tile_shape[a] as usize) as u64)
            .collect(),
        tile_shape,
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
        downsample: None,
    };
    let mut spec = TableSpec::new(0, 2048, "../../", "label_id", columns, spatial_index);
    spec.region = Some(format!("../../labels/{}", args.layer));
    let index = |c: &str| schema.index_of(c).expect("a fit column");
    let centre = [index("center_z"), index("center_y"), index("center_x")];
    let mean_r = index("mean_r");
    let measured: Vec<usize> = measurements.iter().map(|m| index(m)).collect();
    // isotropic grid: the diameter in µm is along x
    let micrometres = voxel_size[2];
    let staged = work.join("table");
    finalize_object_table(
        env,
        KEPT,
        kept_phase,
        volume,
        schema.clone(),
        &staged,
        work,
        spec,
        |row| {
            let mut values = vec![ObjectValue::U64(shape_label(row.at(), volume))];
            for (a, &c) in centre.iter().enumerate() {
                values.push(ObjectValue::F32(
                    row.f64(c)?.clamp(0.0, (volume[a] - 1) as f64) as f32,
                ));
            }
            values.push(ObjectValue::F32(
                (2.0 * row.f64(mean_r)? * micrometres) as f32,
            ));
            for &m in &measured {
                values.push(ObjectValue::F32(row.f64(m)? as f32));
            }
            Ok(Some(values))
        },
    )?;
    publish_label(
        pyramid,
        LabelPublication {
            name: &args.layer,
            level0: &work
                .join("fit")
                .join(format!("level{}", label_image.index())),
            image_label: None,
            table: Some(&staged),
            replace: true,
            pyramid_blocks: Vec::new(),
        },
        args.workers,
    )?;
    eprintln!("published labels/{0} and tables/{0}", args.layer);
    Ok(())
}

fn write_csv(path: &Path, pass: &Pass, args: &Args, radius: f64, volume: [usize; 3]) -> Result<()> {
    let mut f = fs::File::create(path).map_err(Error::backend)?;
    let names: Vec<String> = pass
        .schema
        .columns()
        .iter()
        .map(|c| c.name().to_owned())
        .collect();
    writeln!(f, "label_id,z,y,x,status,{}", names.join(",")).map_err(Error::backend)?;
    for o in &pass.objects {
        let status = if o.kept {
            "kept"
        } else if o.accepted {
            "duplicate"
        } else {
            reason(o, &pass.schema, args, radius)
        };
        let values: Vec<String> = o
            .values
            .iter()
            .map(|v| match v {
                Value::F64(x) => format!("{x:.4}"),
                Value::U64(x) => x.to_string(),
            })
            .collect();
        writeln!(
            f,
            "{},{},{},{},{status},{}",
            if o.kept { shape_label(o.at, volume) } else { 0 },
            o.at[0],
            o.at[1],
            o.at[2],
            values.join(",")
        )
        .map_err(Error::backend)?;
    }
    Ok(())
}

/// The first acceptance test a rejected fit fails, for the report only; the
/// decision itself was the planned filter's.
fn reason(o: &Object, schema: &Schema, args: &Args, radius: f64) -> &'static str {
    let v = |c: &str| o.f(schema, c);
    if v("strong_margin") < 0.0 {
        "weak"
    } else if v("ring_depth") < args.min_ring_depth {
        "lumen"
    } else if v("coverage") < args.min_coverage {
        "coverage"
    } else if v("roughness") > args.max_roughness {
        "rough"
    } else if v("mean_r") < 0.6 * radius || v("mean_r") > 1.6 * radius {
        "size"
    } else {
        "contrast"
    }
}

/// Per-ray radii of the kept objects: one row per object, directions in a
/// header line.
fn write_rays(path: &Path, kept: &[&Object], rays: &RaySet, volume: [usize; 3]) -> Result<()> {
    let mut f = fs::File::create(path).map_err(Error::backend)?;
    let dirs: Vec<String> = rays
        .directions()
        .iter()
        .map(|d| format!("{:.6} {:.6} {:.6}", d[0], d[1], d[2]))
        .collect();
    writeln!(f, "# directions z y x: {}", dirs.join(";")).map_err(Error::backend)?;
    writeln!(f, "label_id,center_z,center_y,center_x,radii...").map_err(Error::backend)?;
    for o in kept {
        let radii: Vec<String> = o.shape.radii.iter().map(|r| format!("{r:.2}")).collect();
        writeln!(
            f,
            "{},{:.2},{:.2},{:.2},{}",
            shape_label(o.at, volume),
            o.shape.center[0],
            o.shape.center[1],
            o.shape.center[2],
            radii.join(",")
        )
        .map_err(Error::backend)?;
    }
    Ok(())
}

/// POST /v1/datasets/rescan, so a running newvolim drops its cached layer list.
fn rescan_viewer(host: &str) {
    use std::io::Read;
    let result = (|| -> std::io::Result<String> {
        let mut s = std::net::TcpStream::connect(host)?;
        s.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;
        let name = host.split(':').next().unwrap_or(host);
        write!(s, "POST /v1/datasets/rescan HTTP/1.1\r\nHost: {name}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
        let mut reply = String::new();
        s.read_to_string(&mut reply)?;
        Ok(reply.lines().next().unwrap_or("").to_owned())
    })();
    match result {
        Ok(status) => eprintln!("viewer {host} rescan: {status}"),
        Err(error) => eprintln!("viewer {host} rescan failed: {error}"),
    }
}
