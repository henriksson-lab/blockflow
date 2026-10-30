// SPDX-License-Identifier: MIT

//! Per-movie DIC calibration from the eggshell, batch-able over recordings.
//!
//! For each recording (an OME-Zarr DIC image group), on a few frames spread
//! over the movie: the eggshell is fitted in every plane
//! (`blockflow::ops::eggshell::fit_shell_plane`), the planes are combined into
//! one ellipsoid per frame, and the shell's edge contrast around the equatorial
//! planes is fitted with `I_edge(φ) = a + b·cos(φ − θ)` (and once more with a
//! `cos 2φ` term). Independently, the gradient moments inside the shell
//! (`blockflow::ops::dic::GradientMoments`) give a shell-free θ. The results go
//! to `<image>/calibration/dic/{summary.json,frames.csv,planes.csv}`, to the
//! image group's `rozt:imageMetadata`, optionally to `labels/eggshell`, and one
//! line per recording to the `--summary` TSV. See the example's README.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use blockflow::image_metadata::{
    self, ImageMetadataDocument, ImageMetadataProperty, ImageMetadataProvenance, SCHEMA_VERSION,
};
use blockflow::ops::dic::{angle_difference, dic_relief_scale, fit_cosine, GradientMoments};
use blockflow::ops::eggshell::{
    auto_seed, ellipsoid_mask, fit_ellipsoid, fit_shell_plane, Ellipsoid, PlaneEllipse, ShellParams,
};
use blockflow::time_series::{LabelStack, TimeSeriesImage};
use blockflow::{Error, Result};
use clap::Parser;
use ndarray::{Array3, Axis};

#[path = "../dic_cli.rs"]
mod dic_cli;
use dic_cli::*;

#[derive(Debug, Parser)]
#[command(
    name = "dic_calibrate",
    about = "Eggshell fit and DIC shear calibration (θ, a, b) per recording, batch-able"
)]
struct Args {
    /// DIC image groups (`rec.ome.zarr/DIC`). Repeatable.
    #[arg(long)]
    image: Vec<PathBuf>,
    /// Also every `<root>/*/<channel>` image group.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Channel group name under each recording, with --root.
    #[arg(long, default_value = "DIC")]
    channel_group: String,
    /// Channel index inside the image.
    #[arg(long, default_value_t = 0)]
    channel: usize,
    /// How many frames, evenly spread over the movie.
    #[arg(long, default_value_t = 8)]
    frames: usize,
    /// Explicit frame indices instead (comma separated).
    #[arg(long, value_delimiter = ',')]
    frame_list: Vec<usize>,
    /// A point inside the embryo, `x,y` level-0 pixels. Default: the centroid of
    /// `tracking/<layer>/detections.csv` when present, else the deepest textured point.
    #[arg(long, value_delimiter = ',')]
    seed: Vec<f64>,
    /// Annotation layer used for the default seed.
    #[arg(long, default_value = "endrov")]
    layer: String,
    /// Planes within this fraction of the z semi-axis of the equator are
    /// used for the contrast fit.
    #[arg(long, default_value_t = 0.45)]
    equator_band: f64,
    /// Frames whose shell relief b is below this fraction of the median over
    /// frames are left out of the movie-level fit (no shell, e.g. hatched).
    #[arg(long, default_value_t = 0.35)]
    min_relative_b: f64,
    /// Minimum inlier fraction of a plane's shell fit to use it.
    #[arg(long, default_value_t = 0.5)]
    min_inliers: f64,
    /// Derivative scale of the gradient-moment estimator, µm.
    #[arg(long, default_value_t = 0.2)]
    gradient_sigma_um: f64,
    /// Write `labels/eggshell` (the shell interior, every frame, nearest calibrated frame's ellipsoid).
    #[arg(long)]
    mask_labels: bool,
    /// Replace an existing `labels/eggshell`.
    #[arg(long)]
    overwrite: bool,
    /// Do not touch the image group's `rozt:imageMetadata`.
    #[arg(long)]
    no_attributes: bool,
    /// One TSV line per recording.
    #[arg(long)]
    summary: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!("dic_calibrate: build with --release");
        return ExitCode::FAILURE;
    }
    let args = Args::parse();
    let mut images = args.image.clone();
    if let Some(root) = &args.root {
        match std::fs::read_dir(root) {
            Ok(entries) => {
                let mut found: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path().join(&args.channel_group))
                    .filter(|p| p.join("zarr.json").is_file())
                    .collect();
                found.sort();
                images.extend(found);
            }
            Err(e) => {
                eprintln!("dic_calibrate: {}: {e}", root.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if images.is_empty() {
        eprintln!("dic_calibrate: give --image or --root");
        return ExitCode::FAILURE;
    }
    let mut lines = vec![
        "image\tstatus\ttheta_deg\ttheta_spread_deg\ta\tb\tb2\ttheta_gradient_deg\testimator_difference_deg\tgradient_anisotropy\tshell_a_um\tshell_b_um\tshell_c_um\tshell_angle_deg\tshell_drift_um\tshell_drift_xy_um\tdetail"
            .to_string(),
    ];
    let mut failed = false;
    for image in &images {
        let started = std::time::Instant::now();
        match calibrate(image, &args) {
            Ok(c) => {
                eprintln!(
                    "{}: θ={:.1}° (±{:.1}) a={:.4} b={:.4} b2={:.4} [2-harmonic θ={:.1}° b={:.4} θ2={:.1}°] | gradient θ={:.1}° (Δ {:.1}°, anisotropy {:.2}) | shell {:.1}×{:.1}×{:.1} µm at {:.1}° [{:.0} s]",
                    image.display(),
                    c.theta_deg,
                    c.theta_spread_deg,
                    c.a,
                    c.b,
                    c.b2,
                    c.theta_2h_deg,
                    c.b_2h,
                    c.theta2_deg,
                    c.theta_gradient_deg,
                    c.estimator_difference_deg,
                    c.gradient_anisotropy,
                    c.shell.a,
                    c.shell.b,
                    c.shell.c,
                    c.shell.angle_deg,
                    started.elapsed().as_secs_f64()
                );
                lines.push(format!(
                    "{}\tok\t{:.2}\t{:.2}\t{:.5}\t{:.5}\t{:.5}\t{:.2}\t{:.2}\t{:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t",
                    image.display(),
                    c.theta_deg,
                    c.theta_spread_deg,
                    c.a,
                    c.b,
                    c.b2,
                    c.theta_gradient_deg,
                    c.estimator_difference_deg,
                    c.gradient_anisotropy,
                    c.shell.a,
                    c.shell.b,
                    c.shell.c,
                    c.shell.angle_deg,
                    c.shell_drift_um,
                    c.shell_drift_xy_um
                ));
            }
            Err(e) => {
                failed = true;
                eprintln!("{}: {e}", image.display());
                lines.push(format!(
                    "{}\terror{}\t{e}",
                    image.display(),
                    "\t".repeat(15)
                ));
            }
        }
    }
    if let Some(path) = &args.summary {
        if let Err(e) = std::fs::write(path, lines.join("\n") + "\n") {
            eprintln!("dic_calibrate: {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

struct PlaneRow {
    t: usize,
    z: usize,
    fit: Option<blockflow::ops::eggshell::ShellPlaneFit>,
    medium: f64,
    error: Option<String>,
}

struct FrameResult {
    record: FrameRecord,
    planes: Vec<PlaneRow>,
    ellipsoid: Option<Ellipsoid>,
    // samples for the movie-level fit: (phi, even/medium, odd/medium)
    samples: Vec<(f64, f64, f64)>,
    moments: GradientMoments,
    relief: f64,
}

fn shell_record(e: &Ellipsoid) -> ShellRecord {
    ShellRecord {
        cx: e.cx,
        cy: e.cy,
        cz: e.cz,
        a: e.a,
        b: e.b,
        c: e.c,
        angle_deg: e.angle.to_degrees(),
    }
}

fn calibrate(image_path: &Path, args: &Args) -> Result<Calibration> {
    let image = TimeSeriesImage::open(image_path)?;
    let spacing = image.spacing(0)?;
    let (dz, px) = (spacing[0], spacing[2]);
    let n_frames = image.n_frames();
    let frames: Vec<usize> = if !args.frame_list.is_empty() {
        args.frame_list
            .iter()
            .copied()
            .filter(|&t| t < n_frames)
            .collect()
    } else {
        let k = args.frames.max(1).min(n_frames);
        (0..k)
            .map(|i| ((i as f64 + 0.5) * n_frames as f64 / k as f64) as usize)
            .collect()
    };
    let params = ShellParams::for_pixel(px)?;
    let seed: [f64; 2] = if args.seed.len() == 2 {
        [args.seed[0], args.seed[1]]
    } else if let Some(ann) = read_annotations(image_path, &args.layer)? {
        let n = ann.len().max(1) as f64;
        [
            ann.iter().map(|a| a.x).sum::<f64>() / n,
            ann.iter().map(|a| a.y).sum::<f64>() / n,
        ]
    } else {
        let v = read_frame(&image, args.channel, frames[0])?;
        let mid = v.len_of(Axis(0)) / 2;
        auto_seed(v.index_axis(Axis(0), mid), &params)?
    };
    let jobs = if args.jobs == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        args.jobs
    };
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Result<FrameResult>>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(frames.len()).max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&t) = frames.get(i) else { break };
                let r = calibrate_frame(&image, args, t, seed, &params, dz, px);
                results.lock().unwrap().push(r);
            });
        }
    });
    let mut results: Vec<FrameResult> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    results.sort_by_key(|r| r.record.t);
    // frames without a shell (hatched, empty field) have b ≈ 0: keep them out
    let typical_b = median(&results.iter().map(|r| r.record.b).collect::<Vec<_>>());
    for r in &mut results {
        r.record.accepted = r.ellipsoid.is_some()
            && r.record.b.is_finite()
            && r.record.b >= args.min_relative_b * typical_b;
    }
    let all_frames = results.len();
    let all_records: Vec<FrameRecord> = results.iter().map(|r| r.record.clone()).collect();
    results.retain(|r| r.record.accepted);
    if results.len() < all_frames {
        eprintln!(
            "{}: {} of {all_frames} frames have no usable shell and are left out",
            image_path.display(),
            all_frames - results.len()
        );
    }

    // movie-level fits over the pooled samples of every frame
    let pooled: Vec<(f64, f64, f64)> = results
        .iter()
        .flat_map(|r| r.samples.iter().copied())
        .collect();
    if pooled.len() < 50 {
        return Err(Error::invalid(format!(
            "only {} shell-edge samples over {} frames; no usable shell",
            pooled.len(),
            frames.len()
        )));
    }
    let phi: Vec<f64> = pooled.iter().map(|s| s.0).collect();
    let even: Vec<f64> = pooled.iter().map(|s| s.1).collect();
    let odd: Vec<f64> = pooled.iter().map(|s| s.2).collect();
    let fit = fit_cosine(&phi, &even, false)?;
    let fit2 = fit_cosine(&phi, &even, true)?;
    let fit_odd = fit_cosine(&phi, &odd, false)?;
    let mut moments = GradientMoments::default();
    for r in &results {
        moments.merge(&r.moments);
    }
    let grad = moments.estimate()?;
    let frame_thetas: Vec<f64> = results
        .iter()
        .map(|r| r.record.theta_deg.to_radians())
        .filter(|v| v.is_finite())
        .collect();
    let (_, spread) = circular_centre(&frame_thetas);
    let medium = median(
        &results
            .iter()
            .map(|r| r.record.medium_level)
            .collect::<Vec<_>>(),
    );
    let shells: Vec<Ellipsoid> = results.iter().filter_map(|r| r.ellipsoid).collect();
    if shells.is_empty() {
        return Err(Error::invalid("no frame gave a shell ellipsoid"));
    }
    let med = |f: &dyn Fn(&Ellipsoid) -> f64| median(&shells.iter().map(f).collect::<Vec<_>>());
    let (angle, _) = {
        let doubled: Vec<f64> = shells.iter().map(|e| 2.0 * e.angle).collect();
        let (c, m) = circular_centre(&doubled);
        (c / 2.0, m / 2.0)
    };
    let shell = Ellipsoid {
        cx: med(&|e| e.cx),
        cy: med(&|e| e.cy),
        cz: med(&|e| e.cz),
        a: med(&|e| e.a),
        b: med(&|e| e.b),
        c: med(&|e| e.c),
        angle,
    };
    let drift = shells
        .iter()
        .map(|e| {
            ((e.cx - shell.cx).powi(2) + (e.cy - shell.cy).powi(2) + (e.cz - shell.cz).powi(2))
                .sqrt()
        })
        .fold(0.0, f64::max);
    let drift_xy = shells
        .iter()
        .map(|e| ((e.cx - shell.cx).powi(2) + (e.cy - shell.cy).powi(2)).sqrt())
        .fold(0.0, f64::max);
    let calibration = Calibration {
        image: image_path.display().to_string(),
        frames: frames.clone(),
        pixel_um: px,
        z_step_um: dz,
        theta_deg: fit.theta.to_degrees(),
        theta_spread_deg: spread.to_degrees(),
        b: fit.b,
        a: fit.a,
        b_raw: fit.b * medium,
        a_raw: fit.a * medium,
        b2: fit2.b2.unwrap_or(f64::NAN),
        theta_2h_deg: fit2.theta.to_degrees(),
        b_2h: fit2.b,
        theta2_deg: fit2.theta2.unwrap_or(f64::NAN).to_degrees(),
        odd_theta_deg: fit_odd.theta.to_degrees(),
        odd_b: fit_odd.b,
        theta_gradient_deg: grad.theta.to_degrees(),
        orientation_gradient_deg: grad.orientation.to_degrees(),
        gradient_anisotropy: grad.anisotropy,
        gradient_skewness: grad.skewness,
        estimator_difference_deg: angle_difference(fit.theta, grad.theta).to_degrees(),
        relief_scale: median(&results.iter().map(|r| r.relief).collect::<Vec<_>>()),
        medium_level: medium,
        shell: shell_record(&shell),
        shell_drift_um: drift,
        shell_drift_xy_um: drift_xy,
        per_frame: all_records,
    };
    write_outputs(image_path, &calibration, &results)?;
    if !args.no_attributes {
        write_attributes(image_path, &calibration)?;
    }
    if args.mask_labels {
        write_mask(image_path, &image, &results, args.overwrite, jobs)?;
    }
    Ok(calibration)
}

fn calibrate_frame(
    image: &TimeSeriesImage,
    args: &Args,
    t: usize,
    seed: [f64; 2],
    params: &ShellParams,
    dz: f64,
    px: f64,
) -> Result<FrameResult> {
    let volume = read_frame(image, args.channel, t)?;
    let nz = volume.len_of(Axis(0));
    let mut planes = Vec::with_capacity(nz);
    for z in 0..nz {
        let plane = volume.index_axis(Axis(0), z);
        let medium = median(&plane.iter().step_by(7).copied().collect::<Vec<_>>());
        match fit_shell_plane(plane, seed, params) {
            Ok(fit) => planes.push(PlaneRow {
                t,
                z,
                fit: Some(fit),
                medium,
                error: None,
            }),
            Err(e) => planes.push(PlaneRow {
                t,
                z,
                fit: None,
                medium,
                error: Some(e.to_string()),
            }),
        }
    }
    let plane_ellipses: Vec<PlaneEllipse> = planes
        .iter()
        .filter_map(|p| {
            let fit = p.fit.as_ref()?;
            Some(PlaneEllipse {
                z: p.z as f64 * dz,
                ellipse: fit.ellipse.scaled(px),
                weight: if fit.inlier_fraction >= args.min_inliers {
                    fit.inlier_fraction
                } else {
                    0.0
                },
            })
        })
        .collect();
    let ellipsoid = fit_ellipsoid(&plane_ellipses).ok();
    let mut samples = Vec::new();
    let mut used = 0;
    let mut moments = GradientMoments::default();
    let mut relief = f64::NAN;
    if let Some(e) = &ellipsoid {
        for p in &planes {
            let Some(fit) = &p.fit else { continue };
            let zum = p.z as f64 * dz;
            let in_band = !e.c.is_finite() || (zum - e.cz).abs() <= args.equator_band * e.c;
            if !in_band || fit.inlier_fraction < args.min_inliers {
                continue;
            }
            used += 1;
            for s in fit.samples.iter().filter(|s| s.inlier) {
                samples.push((s.phi, s.even / p.medium, s.odd / p.medium));
            }
        }
        // gradient moments and relief inside the shell (shrunk), every plane it cuts
        let shape = [nz, volume.len_of(Axis(1)), volume.len_of(Axis(2))];
        let mask = ellipsoid_mask(shape, [dz, px, px], e, 0.9);
        for z in 0..nz {
            let m = mask.index_axis(Axis(0), z);
            if !m.iter().any(|&v| v) {
                continue;
            }
            moments.add_plane(
                volume.index_axis(Axis(0), z),
                Some(m),
                args.gradient_sigma_um / px,
            )?;
        }
        relief = dic_relief_scale(volume.view(), Some(mask.view()), 0.0, 3).unwrap_or(f64::NAN);
    }
    let medium = median(&planes.iter().map(|p| p.medium).collect::<Vec<_>>());
    let nan = f64::NAN;
    let mut record = FrameRecord {
        t,
        shell: ellipsoid.as_ref().map(shell_record),
        planes_fitted: planes.iter().filter(|p| p.fit.is_some()).count(),
        planes_used: used,
        accepted: false,
        samples: samples.len(),
        theta_deg: nan,
        theta_se_deg: nan,
        a: nan,
        b: nan,
        b2: nan,
        theta_2h_deg: nan,
        theta2_deg: nan,
        residual: nan,
        odd_theta_deg: nan,
        odd_b: nan,
        theta_gradient_deg: nan,
        orientation_gradient_deg: nan,
        gradient_anisotropy: nan,
        gradient_skewness: nan,
        medium_level: medium,
        relief_scale: relief,
    };
    if samples.len() >= 50 {
        let phi: Vec<f64> = samples.iter().map(|s| s.0).collect();
        let even: Vec<f64> = samples.iter().map(|s| s.1).collect();
        let odd: Vec<f64> = samples.iter().map(|s| s.2).collect();
        if let Ok(f) = fit_cosine(&phi, &even, false) {
            record.theta_deg = f.theta.to_degrees();
            record.theta_se_deg = f.theta_se.to_degrees();
            record.a = f.a;
            record.b = f.b;
            record.residual = f.residual;
        }
        if let Ok(f) = fit_cosine(&phi, &even, true) {
            record.b2 = f.b2.unwrap_or(nan);
            record.theta_2h_deg = f.theta.to_degrees();
            record.theta2_deg = f.theta2.unwrap_or(nan).to_degrees();
        }
        if let Ok(f) = fit_cosine(&phi, &odd, false) {
            record.odd_theta_deg = f.theta.to_degrees();
            record.odd_b = f.b;
        }
    }
    if let Ok(g) = moments.estimate() {
        record.theta_gradient_deg = g.theta.to_degrees();
        record.orientation_gradient_deg = g.orientation.to_degrees();
        record.gradient_anisotropy = g.anisotropy;
        record.gradient_skewness = g.skewness;
    }
    Ok(FrameResult {
        record,
        planes,
        ellipsoid,
        samples,
        moments,
        relief,
    })
}

fn write_outputs(image_path: &Path, c: &Calibration, results: &[FrameResult]) -> Result<()> {
    let dir = image_path.join("calibration").join("dic");
    std::fs::create_dir_all(&dir).map_err(Error::backend)?;
    let json = serde_json::to_string_pretty(c).map_err(Error::backend)?;
    std::fs::write(calibration_path(image_path), json).map_err(Error::backend)?;
    let header = [
        "t",
        "accepted",
        "planes_fitted",
        "planes_used",
        "samples",
        "theta_deg",
        "theta_se_deg",
        "a",
        "b",
        "b2",
        "theta_2h_deg",
        "theta2_deg",
        "residual",
        "odd_theta_deg",
        "odd_b",
        "theta_gradient_deg",
        "orientation_gradient_deg",
        "gradient_anisotropy",
        "gradient_skewness",
        "medium_level",
        "relief_scale",
        "shell_cx_um",
        "shell_cy_um",
        "shell_cz_um",
        "shell_a_um",
        "shell_b_um",
        "shell_c_um",
        "shell_angle_deg",
    ];
    let rows: Vec<Vec<String>> = c
        .per_frame
        .iter()
        .map(|f| {
            let s = f.shell;
            let sv = |g: &dyn Fn(&ShellRecord) -> f64| fmt(s.as_ref().map(g).unwrap_or(f64::NAN));
            vec![
                f.t.to_string(),
                u8::from(f.accepted).to_string(),
                f.planes_fitted.to_string(),
                f.planes_used.to_string(),
                f.samples.to_string(),
                fmt(f.theta_deg),
                fmt(f.theta_se_deg),
                fmt(f.a),
                fmt(f.b),
                fmt(f.b2),
                fmt(f.theta_2h_deg),
                fmt(f.theta2_deg),
                fmt(f.residual),
                fmt(f.odd_theta_deg),
                fmt(f.odd_b),
                fmt(f.theta_gradient_deg),
                fmt(f.orientation_gradient_deg),
                fmt(f.gradient_anisotropy),
                fmt(f.gradient_skewness),
                fmt(f.medium_level),
                fmt(f.relief_scale),
                sv(&|s| s.cx),
                sv(&|s| s.cy),
                sv(&|s| s.cz),
                sv(&|s| s.a),
                sv(&|s| s.b),
                sv(&|s| s.c),
                sv(&|s| s.angle_deg),
            ]
        })
        .collect();
    write_csv(&dir.join("frames.csv"), &header, &rows)?;
    let header = [
        "t",
        "z",
        "status",
        "cx_px",
        "cy_px",
        "a_px",
        "b_px",
        "angle_deg",
        "inlier_fraction",
        "coverage",
        "line_noise",
        "contrast_snr",
        "medium_level",
        "theta_deg",
        "a",
        "b",
    ];
    let mut rows = Vec::new();
    for r in results {
        for p in &r.planes {
            match &p.fit {
                Some(fit) => {
                    let inl: Vec<_> = fit.samples.iter().filter(|s| s.inlier).collect();
                    let (phi, even): (Vec<f64>, Vec<f64>) =
                        inl.iter().map(|s| (s.phi, s.even / p.medium)).unzip();
                    let cos = fit_cosine(&phi, &even, false).ok();
                    let e = fit.ellipse;
                    rows.push(vec![
                        p.t.to_string(),
                        p.z.to_string(),
                        "ok".into(),
                        fmt(e.cx),
                        fmt(e.cy),
                        fmt(e.a),
                        fmt(e.b),
                        fmt(e.angle.to_degrees()),
                        fmt(fit.inlier_fraction),
                        fmt(fit.coverage),
                        fmt(fit.line_noise),
                        fmt(fit.contrast_snr),
                        fmt(p.medium),
                        fmt(cos.map(|c| c.theta.to_degrees()).unwrap_or(f64::NAN)),
                        fmt(cos.map(|c| c.a).unwrap_or(f64::NAN)),
                        fmt(cos.map(|c| c.b).unwrap_or(f64::NAN)),
                    ]);
                }
                None => {
                    let mut row = vec![p.t.to_string(), p.z.to_string(), "failed".into()];
                    row.extend(std::iter::repeat_n("NaN".to_string(), 9));
                    row.push(fmt(p.medium));
                    row.extend(std::iter::repeat_n("NaN".to_string(), 3));
                    let _ = &p.error;
                    rows.push(row);
                }
            }
        }
    }
    write_csv(&dir.join("planes.csv"), &header, &rows)
}

fn write_attributes(image_path: &Path, c: &Calibration) -> Result<()> {
    let current = image_metadata::read_image_metadata(image_path)?;
    let mut properties = current.document.properties;
    let mut float =
        |key: &str, value: f64, display: &str, unit: Option<&str>, description: &str| {
            if value.is_finite() {
                properties.insert(
                    key.into(),
                    ImageMetadataProperty::Float {
                        value,
                        display_name: Some(display.into()),
                        unit: unit.map(Into::into),
                        description: Some(description.into()),
                    },
                );
            }
        };
    float(
        "dic.theta_deg",
        c.theta_deg,
        "DIC shear θ (shell)",
        Some("degree"),
        "bright-edge angle from the eggshell fit a + b cos(φ − θ); image coords, x right, y down",
    );
    float(
        "dic.b",
        c.b,
        "DIC relief b",
        None,
        "shell relief strength, relative to the medium level",
    );
    float(
        "dic.a",
        c.a,
        "DIC symmetric term a",
        None,
        "angle-independent shell contrast, relative to the medium level",
    );
    float(
        "dic.b2",
        c.b2,
        "DIC cos 2φ term",
        None,
        "second-harmonic amplitude of the shell contrast, relative",
    );
    float(
        "dic.theta_gradient_deg",
        c.theta_gradient_deg,
        "DIC shear θ (gradients)",
        Some("degree"),
        "shell-free bright-edge angle from gradient moments inside the shell",
    );
    float(
        "dic.estimator_difference_deg",
        c.estimator_difference_deg,
        "DIC θ shell − gradient",
        Some("degree"),
        "agreement of the two estimators",
    );
    float(
        "dic.relief_scale",
        c.relief_scale,
        "DIC relief scale",
        None,
        "1.4826·MAD of I − median inside the shell, raw units",
    );
    float(
        "eggshell.cx_um",
        c.shell.cx,
        "Eggshell centre x",
        Some("micrometer"),
        "ellipsoid centre",
    );
    float(
        "eggshell.cy_um",
        c.shell.cy,
        "Eggshell centre y",
        Some("micrometer"),
        "ellipsoid centre",
    );
    float(
        "eggshell.cz_um",
        c.shell.cz,
        "Eggshell centre z",
        Some("micrometer"),
        "ellipsoid centre",
    );
    float(
        "eggshell.a_um",
        c.shell.a,
        "Eggshell semi-axis a (AP)",
        Some("micrometer"),
        "long in-plane semi-axis",
    );
    float(
        "eggshell.b_um",
        c.shell.b,
        "Eggshell semi-axis b",
        Some("micrometer"),
        "short in-plane semi-axis",
    );
    float(
        "eggshell.c_um",
        c.shell.c,
        "Eggshell semi-axis c (z)",
        Some("micrometer"),
        "semi-axis along z",
    );
    float(
        "eggshell.ap_angle_deg",
        c.shell.angle_deg,
        "AP axis angle",
        Some("degree"),
        "angle of the long in-plane axis (sign of AP not determined)",
    );
    let updated_at_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(Error::backend)?
        .as_millis() as u64;
    let document = ImageMetadataDocument {
        schema_version: SCHEMA_VERSION,
        properties,
        provenance: Some(ImageMetadataProvenance {
            writer: "blockflow-celegans-4d dic_calibrate".into(),
            updated_at_ms,
        }),
    };
    image_metadata::replace_image_metadata(image_path, Some(&current.revision), &document).map(drop)
}

/// `labels/eggshell`: value 1 inside the shell at every frame and level, using
/// the calibrated frame nearest in time.
fn write_mask(
    image_path: &Path,
    image: &TimeSeriesImage,
    results: &[FrameResult],
    overwrite: bool,
    jobs: usize,
) -> Result<()> {
    let root = image_path.join("labels").join("eggshell");
    if root.exists() {
        if !overwrite {
            return Err(Error::invalid(format!(
                "{} exists; pass --overwrite",
                root.display()
            )));
        }
        std::fs::remove_dir_all(&root).map_err(Error::backend)?;
    }
    let shells: Vec<(usize, Ellipsoid)> = results
        .iter()
        .filter_map(|r| r.ellipsoid.map(|e| (r.record.t, e)))
        .collect();
    let tmp = image_path.join("labels").join(".eggshell.tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    let stack = LabelStack::create(&tmp, image, [256, 256], "../../")?;
    let n = image.n_frames();
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= n {
                    break;
                }
                let shell = shells
                    .iter()
                    .min_by_key(|(ts, _)| ts.abs_diff(t))
                    .map(|(_, e)| *e)
                    .unwrap();
                let outcome = (|| -> Result<()> {
                    for level in 0..stack.n_levels() {
                        let shape = stack.frame_shape(level)?;
                        // physical voxel k at k·scale, as the level's own transform states (translation 0 here)
                        let sp = image.spacing(level)?;
                        let mask = ellipsoid_mask(shape, sp, &shell, 1.0);
                        let labels: Array3<u64> = mask.mapv(u64::from);
                        stack.write_plane(level, t, &labels)?;
                    }
                    Ok(())
                })();
                if let Err(e) = outcome {
                    *failure.lock().unwrap() = Some(e);
                    break;
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    std::fs::rename(&tmp, &root).map_err(Error::backend)?;
    blockflow::label_pyramid::refresh_label_registry(&image_path.join("labels")).map(drop)
}
