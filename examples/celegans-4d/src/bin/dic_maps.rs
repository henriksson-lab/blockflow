// SPDX-License-Identifier: MIT

//! DIC maps from a calibrated recording, their validation against hand
//! annotations, and (optionally) the maps as sibling image groups for ROZT.
//!
//! Per frame: `(I − median)/relief_scale` (`dic_normalise_plane`), the phase-like
//! reconstruction along the calibrated shear (`dic_integrate_plane`), the
//! smoothness `−ln E` (`dic_texture_volume`) and the nucleus-likeness
//! (`centre_surround_volume`). Needs `dic_calibrate`'s
//! `<image>/calibration/dic/summary.json`.
//!
//! `--validate`: at every annotated keyframe, each smoothness variant
//! (raw / phase × energy / isotropic) is read at the nucleus centre and at
//! random points inside the shell of the same plane → ROC AUC; the
//! disc-vs-ring contrast of the README's mapping check; and the distance from
//! each centre to the nearest local maximum of the nucleus-likeness map, with
//! the same distance from random interior points as the chance level.
//!
//! `--write`: `<recording>/<group>_phase`, `<group>_smooth`, `<group>_nucleus`
//! (uint8, same grid as the DIC, unwritten frames read 0) and a shared
//! `rozt.channelGroup` so ROZT shows them as channels of the DIC dataset.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use blockflow::dtype::Dtype;
use blockflow::ops::dic::{
    centre_surround_volume, dic_integrate_plane, dic_normalise_plane, dic_texture_volume,
    CentreSurround, DicIntegration, DicNormalise, DicTexture, TextureMeasure,
};
use blockflow::ops::eggshell::Ellipsoid;
use blockflow::time_series::TimeSeriesImage;
use blockflow::zarr_env::Compression;
use blockflow::zarr_layout::StorageLayout;
use blockflow::{Error, Result};
use clap::{Parser, ValueEnum};
use ndarray::{Array3, Axis};
use serde_json::{json, Value};
use zarrs::array::{Array as ZarrArray, ArraySubset};
use zarrs::filesystem::FilesystemStore;

#[path = "../dic_cli.rs"]
mod dic_cli;
use dic_cli::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FrameChoice {
    /// Frames with at least one annotated keyframe.
    Keyframes,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Source {
    Raw,
    Phase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Measure {
    Energy,
    Isotropic,
}

#[derive(Debug, Parser)]
#[command(
    name = "dic_maps",
    about = "DIC phase / smoothness / nucleus-likeness maps, validation, ROZT groups"
)]
struct Args {
    #[arg(long)]
    image: PathBuf,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    #[arg(long, value_enum, default_value_t = FrameChoice::Keyframes)]
    frames: FrameChoice,
    /// Explicit frames instead of --frames.
    #[arg(long, value_delimiter = ',')]
    frame_list: Vec<usize>,
    /// Annotation layer (`tracking/<layer>/detections.csv`).
    #[arg(long, default_value = "endrov")]
    layer: String,
    /// Leak length of the phase integration, µm.
    #[arg(long, default_value_t = 10.0)]
    leak_um: f64,
    /// Granule band (inner, outer Gaussian sigma), µm.
    #[arg(long, value_delimiter = ',', default_values_t = vec![0.13, 0.5])]
    band_um: Vec<f64>,
    /// Texture integration sigma, µm.
    #[arg(long, default_value_t = 1.0)]
    window_um: f64,
    /// Smoothing of the texture energy across planes, in planes.
    #[arg(long, default_value_t = 0.0)]
    z_sigma: f64,
    /// Nucleus radii of the centre-surround, µm.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1.5, 2.5, 3.5])]
    radii_um: Vec<f64>,
    /// The smoothness map that is written and feeds the nucleus-likeness.
    #[arg(long, value_enum, default_value_t = Source::Raw)]
    source: Source,
    #[arg(long, value_enum, default_value_t = Measure::Isotropic)]
    measure: Measure,
    #[arg(long)]
    validate: bool,
    #[arg(long)]
    write: bool,
    /// Replace existing map groups.
    #[arg(long)]
    overwrite: bool,
    /// Validation output directory (default `<image>/calibration/dic/validation`).
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Random interior points per annotated plane.
    #[arg(long, default_value_t = 200)]
    negatives: usize,
    #[arg(long, default_value_t = 0)]
    jobs: usize,
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!("dic_maps: build with --release");
        return ExitCode::FAILURE;
    }
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("dic_maps: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Maps {
    normalise: DicNormalise,
    integration: DicIntegration,
    textures: Vec<(String, Source, DicTexture)>,
    centre: CentreSurround,
    main: usize,
}

const VARIANTS: [(&str, Source, Measure); 4] = [
    ("raw_energy", Source::Raw, Measure::Energy),
    ("raw_isotropic", Source::Raw, Measure::Isotropic),
    ("phase_energy", Source::Phase, Measure::Energy),
    ("phase_isotropic", Source::Phase, Measure::Isotropic),
];

fn build_maps(args: &Args, cal: &Calibration) -> Result<Maps> {
    let px = cal.pixel_um;
    if args.band_um.len() != 2 {
        return Err(Error::invalid("--band-um takes two values"));
    }
    let band = (args.band_um[0] / px, args.band_um[1] / px);
    let mut textures = Vec::new();
    let mut main = 0;
    for (k, (name, source, measure)) in VARIANTS.iter().enumerate() {
        let m = match measure {
            Measure::Energy => TextureMeasure::Energy,
            Measure::Isotropic => TextureMeasure::Isotropic,
        };
        // phase is in (relief units × pixels); its floor scales with it
        let floor = if *source == Source::Phase { 1e-4 } else { 1e-6 };
        textures.push((
            name.to_string(),
            *source,
            DicTexture::new(band, args.window_um / px, m, args.z_sigma, floor)?,
        ));
        if *source == args.source && *measure == args.measure {
            main = k;
        }
    }
    Ok(Maps {
        normalise: DicNormalise::new(0.0, cal.relief_scale)?,
        integration: DicIntegration::new(cal.theta_deg.to_radians(), args.leak_um / px)?,
        textures,
        centre: CentreSurround::new(args.radii_um.iter().map(|r| r / px).collect(), 1.6)?,
        main,
    })
}

struct FrameMaps {
    phase: Array3<f64>,
    smooth: Vec<Array3<f64>>,
    nucleus: Array3<f64>,
}

/// The maps of the planes in `planes` (others left zero).
fn compute(
    volume: &Array3<f64>,
    maps: &Maps,
    planes: &[usize],
    all_variants: bool,
) -> Result<FrameMaps> {
    let (nz, ny, nx) = volume.dim();
    let mut norm = Array3::<f64>::zeros((nz, ny, nx));
    let mut phase = Array3::<f64>::zeros((nz, ny, nx));
    // texture z-smoothing reaches neighbouring planes: compute those too
    let reach = maps
        .textures
        .iter()
        .map(|t| t.2.reach_z())
        .max()
        .unwrap_or(0);
    let mut needed = vec![false; nz];
    for &z in planes {
        for zz in z.saturating_sub(reach)..(z + reach + 1).min(nz) {
            needed[zz] = true;
        }
    }
    for z in 0..nz {
        if !needed[z] {
            continue;
        }
        let n = dic_normalise_plane(volume.index_axis(Axis(0), z), &maps.normalise)?;
        phase
            .index_axis_mut(Axis(0), z)
            .assign(&dic_integrate_plane(n.view(), &maps.integration)?);
        norm.index_axis_mut(Axis(0), z).assign(&n);
    }
    let mut smooth = Vec::new();
    for (k, (_, source, params)) in maps.textures.iter().enumerate() {
        if !all_variants && k != maps.main {
            smooth.push(Array3::zeros((0, 0, 0)));
            continue;
        }
        let input = if *source == Source::Phase {
            &phase
        } else {
            &norm
        };
        smooth.push(dic_texture_volume(input.view(), params)?);
    }
    let nucleus = centre_surround_volume(smooth[maps.main].view(), &maps.centre)?;
    Ok(FrameMaps {
        phase,
        smooth,
        nucleus,
    })
}

fn frame_shell(cal: &Calibration, t: usize) -> Ellipsoid {
    let rec = cal
        .per_frame
        .iter()
        .filter(|f| f.accepted)
        .filter_map(|f| f.shell.map(|s| (f.t, s)))
        .min_by_key(|(ft, _)| ft.abs_diff(t))
        .map(|(_, s)| s)
        .unwrap_or(cal.shell);
    Ellipsoid {
        cx: rec.cx,
        cy: rec.cy,
        cz: rec.cz,
        a: rec.a,
        b: rec.b,
        c: rec.c,
        angle: rec.angle_deg.to_radians(),
    }
}

fn run(args: &Args) -> Result<()> {
    let cal = read_calibration(&args.image)?;
    let image = TimeSeriesImage::open(&args.image)?;
    let maps = build_maps(args, &cal)?;
    let annotations = read_annotations(&args.image, &args.layer)?.unwrap_or_default();
    let keyed = keyframes_by_frame(&annotations);
    let frames: Vec<usize> = if !args.frame_list.is_empty() {
        args.frame_list.clone()
    } else {
        match args.frames {
            FrameChoice::Keyframes => keyed.keys().copied().collect(),
            FrameChoice::All => (0..image.n_frames()).collect(),
        }
    };
    if frames.is_empty() {
        return Err(Error::invalid(
            "no frames to process (no keyframes? use --frames all)",
        ));
    }
    let jobs = if args.jobs == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16)
    } else {
        args.jobs
    };
    let writer = if args.write {
        Some(Writer::create(&args.image, &image, args.overwrite)?)
    } else {
        None
    };
    let quant: Mutex<Option<[(f64, f64); 3]>> = Mutex::new(None);
    let records: Mutex<Vec<ValRecord>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<Error>> = Mutex::new(None);
    let nz = image.frame_shape(0)?[0];
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(frames.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&t) = frames.get(i) else { break };
                let outcome = (|| -> Result<()> {
                    let volume = read_frame(&image, args.channel, t)?;
                    let gt: Vec<Annotation> = keyed.get(&t).cloned().unwrap_or_default();
                    let planes: Vec<usize> = if args.write {
                        (0..nz).collect()
                    } else {
                        let mut p: Vec<usize> = gt
                            .iter()
                            .map(|a| a.z.round() as i64)
                            .filter(|&z| z >= 0 && (z as usize) < nz)
                            .map(|z| z as usize)
                            .collect();
                        p.sort();
                        p.dedup();
                        p
                    };
                    if planes.is_empty() {
                        return Ok(());
                    }
                    let fm = compute(&volume, &maps, &planes, args.validate)?;
                    if args.validate && !gt.is_empty() {
                        let shell = frame_shell(&cal, t);
                        let recs =
                            validate_frame(t, &gt, &fm, &maps, &shell, &cal, args.negatives)?;
                        records.lock().unwrap().extend(recs);
                    }
                    if let Some(w) = &writer {
                        let q = {
                            let mut guard = quant.lock().unwrap();
                            *guard.get_or_insert_with(|| quantisation(&fm, maps.main))
                        };
                        w.write(t, &fm, maps.main, &q)?;
                    }
                    Ok(())
                })();
                if let Err(e) = outcome {
                    *failure.lock().unwrap() = Some(Error::invalid(format!("frame {t}: {e}")));
                    break;
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    if let Some(w) = &writer {
        let q = quant.into_inner().unwrap().unwrap_or([(0.0, 1.0); 3]);
        w.finish(&args.image, &q, &maps, &cal)?;
        eprintln!("wrote {} frames into {}", frames.len(), w.names.join(", "));
    }
    if args.validate {
        let mut recs = records.into_inner().unwrap();
        recs.sort_by(|a, b| (a.t, a.kind, a.z).partial_cmp(&(b.t, b.kind, b.z)).unwrap());
        let out = args.out_dir.clone().unwrap_or_else(|| {
            args.image
                .join("calibration")
                .join("dic")
                .join("validation")
        });
        report(&recs, &maps, &out, &cal)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ValRecord {
    t: usize,
    /// 0 = annotated centre, 1 = random interior point
    kind: u8,
    z: usize,
    x: f64,
    y: f64,
    radius_um: f64,
    values: Vec<f64>,
    /// disc (0.6 r) minus ring (1.1–1.6 r) mean of the main smoothness
    contrast: f64,
    nucleus: f64,
    /// distance to the nearest local maximum of the nucleus-likeness, µm
    nearest_max_um: f64,
}

fn local_maxima(
    plane: ndarray::ArrayView2<'_, f64>,
    radius: usize,
    inside: &dyn Fn(usize, usize) -> bool,
) -> Vec<(f64, f64)> {
    let (ny, nx) = plane.dim();
    let mut out = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let v = plane[[y, x]];
            if !(v > 0.0) || !inside(y, x) {
                continue;
            }
            let mut is_max = true;
            'win: for yy in y.saturating_sub(radius)..(y + radius + 1).min(ny) {
                for xx in x.saturating_sub(radius)..(x + radius + 1).min(nx) {
                    let w = plane[[yy, xx]];
                    if w > v || (w == v && (yy, xx) < (y, x)) {
                        is_max = false;
                        break 'win;
                    }
                }
            }
            if is_max {
                out.push((x as f64, y as f64));
            }
        }
    }
    out
}

fn disc_ring(plane: ndarray::ArrayView2<'_, f64>, x: f64, y: f64, r: f64) -> f64 {
    let (ny, nx) = plane.dim();
    let (mut d, mut dn, mut g, mut gn) = (0.0, 0.0, 0.0, 0.0);
    let reach = (1.6 * r).ceil() as i64;
    for yy in (y.round() as i64 - reach)..=(y.round() as i64 + reach) {
        for xx in (x.round() as i64 - reach)..=(x.round() as i64 + reach) {
            if yy < 0 || xx < 0 || yy >= ny as i64 || xx >= nx as i64 {
                continue;
            }
            let dist = ((yy as f64 - y).powi(2) + (xx as f64 - x).powi(2)).sqrt();
            let v = plane[[yy as usize, xx as usize]];
            if dist < 0.6 * r {
                d += v;
                dn += 1.0;
            } else if dist > 1.1 * r && dist < 1.6 * r {
                g += v;
                gn += 1.0;
            }
        }
    }
    if dn > 0.0 && gn > 0.0 {
        d / dn - g / gn
    } else {
        f64::NAN
    }
}

fn validate_frame(
    t: usize,
    gt: &[Annotation],
    fm: &FrameMaps,
    maps: &Maps,
    shell: &Ellipsoid,
    cal: &Calibration,
    negatives: usize,
) -> Result<Vec<ValRecord>> {
    let (nz, ny, nx) = fm.nucleus.dim();
    let px = cal.pixel_um;
    let inner = Ellipsoid {
        a: shell.a * 0.85,
        b: shell.b * 0.85,
        c: shell.c * 0.85,
        ..*shell
    };
    let inside = |z: usize, y: usize, x: usize| {
        inner.radius_of(x as f64 * px, y as f64 * px, z as f64 * cal.z_step_um) < 1.0
    };
    let mut by_plane: BTreeMap<usize, Vec<Annotation>> = BTreeMap::new();
    for a in gt {
        let z = a.z.round();
        if z >= 0.0
            && (z as usize) < nz
            && a.x >= 0.0
            && a.y >= 0.0
            && a.x < nx as f64
            && a.y < ny as f64
        {
            by_plane.entry(z as usize).or_default().push(*a);
        }
    }
    let mut out = Vec::new();
    let mut rng = (t as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut rand = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let max_radius = (1.0 / px).round() as usize;
    for (&z, list) in &by_plane {
        let nplane = fm.nucleus.index_axis(Axis(0), z);
        let maxima = local_maxima(nplane, max_radius, &|y, x| inside(z, y, x));
        let nearest = |x: f64, y: f64| {
            maxima
                .iter()
                .map(|&(mx, my)| ((mx - x).powi(2) + (my - y).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
                * px
        };
        let sample = |x: f64, y: f64| -> Vec<f64> {
            let (xi, yi) = (
                x.round().clamp(0.0, nx as f64 - 1.0) as usize,
                y.round().clamp(0.0, ny as f64 - 1.0) as usize,
            );
            fm.smooth
                .iter()
                .map(|s| {
                    if s.dim().0 == 0 {
                        f64::NAN
                    } else {
                        s[[z, yi, xi]]
                    }
                })
                .collect()
        };
        let main = fm.smooth[maps.main].index_axis(Axis(0), z);
        for a in list {
            let r = a.radius / px;
            out.push(ValRecord {
                t,
                kind: 0,
                z,
                x: a.x,
                y: a.y,
                radius_um: a.radius,
                values: sample(a.x, a.y),
                contrast: disc_ring(main, a.x, a.y, r),
                nucleus: nplane[[a.y.round() as usize, a.x.round() as usize]],
                nearest_max_um: nearest(a.x, a.y),
            });
        }
        let mut got = 0;
        let mut tries = 0;
        while got < negatives && tries < negatives * 200 {
            tries += 1;
            let (x, y) = (rand() * (nx - 1) as f64, rand() * (ny - 1) as f64);
            if !inside(z, y.round() as usize, x.round() as usize) {
                continue;
            }
            got += 1;
            out.push(ValRecord {
                t,
                kind: 1,
                z,
                x,
                y,
                radius_um: f64::NAN,
                values: sample(x, y),
                contrast: f64::NAN,
                nucleus: nplane[[y.round() as usize, x.round() as usize]],
                nearest_max_um: nearest(x, y),
            });
        }
    }
    Ok(out)
}

/// ROC AUC by the rank-sum: P(positive > negative), ties counted half.
fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    let mut all: Vec<(f64, bool)> = pos
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| (v, true))
        .chain(neg.iter().filter(|v| v.is_finite()).map(|&v| (v, false)))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (np, nn) = (
        all.iter().filter(|a| a.1).count() as f64,
        all.iter().filter(|a| !a.1).count() as f64,
    );
    if np == 0.0 || nn == 0.0 {
        return f64::NAN;
    }
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let rank = 0.5 * ((i + 1) + (j + 1)) as f64;
        rank_sum += rank * all[i..=j].iter().filter(|a| a.1).count() as f64;
        i = j + 1;
    }
    (rank_sum - np * (np + 1.0) / 2.0) / (np * nn)
}

fn report(recs: &[ValRecord], maps: &Maps, out: &Path, cal: &Calibration) -> Result<()> {
    let pos: Vec<&ValRecord> = recs.iter().filter(|r| r.kind == 0).collect();
    let neg: Vec<&ValRecord> = recs.iter().filter(|r| r.kind == 1).collect();
    let mut summary = serde_json::Map::new();
    summary.insert("image".into(), json!(cal.image));
    summary.insert("annotated_centres".into(), json!(pos.len()));
    summary.insert("random_points".into(), json!(neg.len()));
    summary.insert(
        "frames".into(),
        json!(recs
            .iter()
            .map(|r| r.t)
            .collect::<std::collections::BTreeSet<_>>()
            .len()),
    );
    let mut aucs = serde_json::Map::new();
    for (k, (name, _, _)) in maps.textures.iter().enumerate() {
        let p: Vec<f64> = pos.iter().map(|r| r.values[k]).collect();
        let n: Vec<f64> = neg.iter().map(|r| r.values[k]).collect();
        aucs.insert(name.clone(), json!(auc(&p, &n)));
    }
    summary.insert("auc_smoothness".into(), Value::Object(aucs));
    let pn: Vec<f64> = pos.iter().map(|r| r.nucleus).collect();
    let nn: Vec<f64> = neg.iter().map(|r| r.nucleus).collect();
    summary.insert("auc_nucleus_likeness".into(), json!(auc(&pn, &nn)));
    let contrast: Vec<f64> = pos
        .iter()
        .map(|r| r.contrast)
        .filter(|v| v.is_finite())
        .collect();
    summary.insert("disc_minus_ring_median".into(), json!(median(&contrast)));
    summary.insert(
        "disc_minus_ring_fraction_positive".into(),
        json!(contrast.iter().filter(|&&c| c > 0.0).count() as f64 / contrast.len().max(1) as f64),
    );
    let dp: Vec<f64> = pos.iter().map(|r| r.nearest_max_um).collect();
    let dn: Vec<f64> = neg.iter().map(|r| r.nearest_max_um).collect();
    let within = |d: &[f64], lim: f64| {
        d.iter().filter(|&&v| v <= lim).count() as f64 / d.len().max(1) as f64
    };
    summary.insert("nearest_max_median_um".into(), json!(median(&dp)));
    summary.insert("nearest_max_median_um_random".into(), json!(median(&dn)));
    summary.insert("nearest_max_within_1um".into(), json!(within(&dp, 1.0)));
    summary.insert(
        "nearest_max_within_1um_random".into(),
        json!(within(&dn, 1.0)),
    );
    summary.insert("nearest_max_within_2um".into(), json!(within(&dp, 2.0)));
    summary.insert(
        "nearest_max_within_2um_random".into(),
        json!(within(&dn, 2.0)),
    );
    let within_r = pos
        .iter()
        .filter(|r| r.nearest_max_um <= r.radius_um)
        .count() as f64
        / pos.len().max(1) as f64;
    summary.insert("nearest_max_within_radius".into(), json!(within_r));
    summary.insert("main_variant".into(), json!(maps.textures[maps.main].0));
    let text = serde_json::to_string_pretty(&Value::Object(summary)).map_err(Error::backend)?;
    std::fs::create_dir_all(out).map_err(Error::backend)?;
    std::fs::write(out.join("summary.json"), &text).map_err(Error::backend)?;
    println!("{text}");
    let mut header: Vec<String> = ["t", "kind", "z", "x", "y", "radius_um"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    header.extend(maps.textures.iter().map(|t| t.0.clone()));
    header.extend(
        ["disc_minus_ring", "nucleus_likeness", "nearest_max_um"]
            .iter()
            .map(|s| s.to_string()),
    );
    let rows: Vec<Vec<String>> = recs
        .iter()
        .map(|r| {
            let mut row = vec![
                r.t.to_string(),
                if r.kind == 0 {
                    "centre".into()
                } else {
                    "random".into()
                },
                r.z.to_string(),
                fmt(r.x),
                fmt(r.y),
                fmt(r.radius_um),
            ];
            row.extend(r.values.iter().map(|&v| fmt(v)));
            row.extend([fmt(r.contrast), fmt(r.nucleus), fmt(r.nearest_max_um)]);
            row
        })
        .collect();
    let header_refs: Vec<&str> = header.iter().map(String::as_str).collect();
    write_csv(&out.join("points.csv"), &header_refs, &rows)
}

// ---------------------------------------------------------------------------
// ROZT groups
// ---------------------------------------------------------------------------

/// (offset, scale) per map: stored = clamp((value − offset)/scale, 0, 255).
fn quantisation(fm: &FrameMaps, main: usize) -> [(f64, f64); 3] {
    let range = |a: &Array3<f64>| {
        let mut v: Vec<f64> = a
            .iter()
            .step_by(13)
            .copied()
            .filter(|x| x.is_finite() && *x != 0.0)
            .collect();
        if v.is_empty() {
            return (0.0, 1.0);
        }
        v.sort_by(f64::total_cmp);
        let lo = v[(v.len() as f64 * 0.005) as usize];
        let hi = v[((v.len() as f64 * 0.995) as usize).min(v.len() - 1)];
        (lo, ((hi - lo) / 255.0).max(1e-12))
    };
    [
        range(&fm.phase),
        range(&fm.smooth[main]),
        range(&fm.nucleus),
    ]
}

struct Writer {
    roots: Vec<PathBuf>,
    names: Vec<String>,
    shape: Vec<usize>,
}

const SUFFIXES: [&str; 3] = ["phase", "smooth", "nucleus"];

impl Writer {
    fn create(image_path: &Path, image: &TimeSeriesImage, overwrite: bool) -> Result<Self> {
        let parent = image_path
            .parent()
            .ok_or_else(|| Error::invalid("the image group has no parent"))?;
        let group = image_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("DIC");
        let [nz, ny, nx] = image.frame_shape(0)?;
        let shape = vec![image.n_frames(), 1, nz, ny, nx];
        let mut roots = Vec::new();
        let mut names = Vec::new();
        for suffix in SUFFIXES {
            let name = format!("{group}_{suffix}");
            let root = parent.join(&name);
            if root.exists() {
                if !overwrite {
                    return Err(Error::invalid(format!(
                        "{} exists; pass --overwrite",
                        root.display()
                    )));
                }
                std::fs::remove_dir_all(&root).map_err(Error::backend)?;
            }
            let level = root.join("0");
            std::fs::create_dir_all(&level).map_err(Error::backend)?;
            let chunk = vec![1, 1, nz, 128.min(ny), 128.min(nx)];
            let metadata = StorageLayout::chunked(chunk, Compression::for_dtype(Dtype::U8))
                .array_metadata(
                    &shape,
                    Dtype::U8,
                    json!(0),
                    Some(json!(["t", "c", "z", "y", "x"])),
                )?;
            std::fs::write(
                level.join("zarr.json"),
                serde_json::to_vec_pretty(&metadata).map_err(Error::backend)?,
            )
            .map_err(Error::backend)?;
            roots.push(root);
            names.push(name);
        }
        Ok(Self {
            roots,
            names,
            shape,
        })
    }

    fn write(&self, t: usize, fm: &FrameMaps, main: usize, q: &[(f64, f64); 3]) -> Result<()> {
        let arrays = [&fm.phase, &fm.smooth[main], &fm.nucleus];
        for ((root, array), (offset, scale)) in self.roots.iter().zip(arrays).zip(q) {
            let bytes: Vec<u8> = array
                .as_standard_layout()
                .iter()
                .map(|&v| {
                    if v.is_finite() {
                        ((v - offset) / scale).round().clamp(1.0, 255.0) as u8
                    } else {
                        0
                    }
                })
                .collect();
            let store =
                std::sync::Arc::new(FilesystemStore::new(root.join("0")).map_err(Error::backend)?);
            let zarray = ZarrArray::open(store, "/").map_err(Error::backend)?;
            let start = vec![t as u64, 0, 0, 0, 0];
            let extent: Vec<u64> = std::iter::once(1u64)
                .chain(self.shape[1..].iter().map(|&n| n as u64))
                .collect();
            let subset =
                ArraySubset::new_with_start_shape(start, extent).map_err(Error::backend)?;
            zarray
                .store_array_subset(&subset, &bytes)
                .map_err(Error::backend)?;
        }
        Ok(())
    }

    /// Group metadata (after the frames, so a scan never sees a half-made group
    /// as finished), and the shared channel group on the source.
    fn finish(
        &self,
        image_path: &Path,
        q: &[(f64, f64); 3],
        maps: &Maps,
        cal: &Calibration,
    ) -> Result<()> {
        let source_json = image_path.join("zarr.json");
        let mut source: Value =
            serde_json::from_slice(&std::fs::read(&source_json).map_err(Error::backend)?)
                .map_err(Error::backend)?;
        let recording = image_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("recording")
            .trim_end_matches(".ome.zarr")
            .to_string();
        let ms = source["attributes"]["ome"]["multiscales"][0].clone();
        let axes = ms["axes"].clone();
        let transforms = ms["datasets"][0]["coordinateTransformations"].clone();
        let descriptions = [
            format!("DIC phase-like reconstruction: Tikhonov-regularised inverse of the shear derivative, θ = {:.1}°, leak {:.1} px", cal.theta_deg, maps.integration.leak_length),
            format!("DIC smoothness −ln E ({}), higher = smoother", maps.textures[maps.main].0),
            "DIC nucleus-likeness: centre-surround of the smoothness, max over radii".to_string(),
        ];
        let colors = ["FFFFFF", "00FF00", "FF00FF"];
        for (k, (root, name)) in self.roots.iter().zip(&self.names).enumerate() {
            let (offset, scale) = q[k];
            let group = json!({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [{"name": name, "axes": axes, "datasets": [{"path": "0", "coordinateTransformations": transforms}]}],
                        "omero": {"channels": [{"label": name, "color": colors[k], "active": true, "window": {"start": 1, "end": 255, "min": 0, "max": 255}}]}
                    },
                    "rozt": {"channelGroup": recording},
                    "blockflow:dicMap": {"description": descriptions[k], "value": "offset + stored · scale (stored 0 = not computed)", "offset": offset, "scale": scale, "source": image_path.file_name().and_then(|n| n.to_str())}
                }
            });
            std::fs::write(
                root.join("zarr.json"),
                serde_json::to_vec_pretty(&group).map_err(Error::backend)?,
            )
            .map_err(Error::backend)?;
        }
        source["attributes"]["rozt"]["channelGroup"] = json!(recording);
        let tmp = image_path.join(".zarr.json.dic_maps.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(&source).map_err(Error::backend)?,
        )
        .map_err(Error::backend)?;
        std::fs::rename(&tmp, &source_json).map_err(Error::backend)?;
        Ok(())
    }
}
