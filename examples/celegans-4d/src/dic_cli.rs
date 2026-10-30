// SPDX-License-Identifier: MIT

//! Shared by the `dic_calibrate` and `dic_maps` binaries (included with
//! `#[path]`, not part of the library): frame reading, annotation points, the
//! calibration summary both binaries exchange, and small statistics.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use blockflow::time_series::TimeSeriesImage;
use blockflow::{Error, Result};
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// One frame, level 0, as `f64` `[z, y, x]`.
pub fn read_frame(image: &TimeSeriesImage, channel: usize, t: usize) -> Result<Array3<f64>> {
    Ok(image.frame(0, channel, t)?.read()?.widened())
}

/// An annotated nucleus position (level-0 voxels, table convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Annotation {
    pub t: usize,
    pub z: f64,
    pub y: f64,
    pub x: f64,
    /// Physical radius, µm.
    pub radius: f64,
    pub keyframe: bool,
}

/// Read `tracking/<layer>/detections.csv` as written by `import_endrov_lineage`.
/// `None` when the file does not exist.
pub fn read_annotations(image: &Path, layer: &str) -> Result<Option<Vec<Annotation>>> {
    let path = image.join("tracking").join(layer).join("detections.csv");
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).map_err(Error::backend)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| Error::invalid(format!("{}: no column {name}", path.display())))
    };
    let (ct, cz, cy, cx, cr, cs) = (
        col("t_index")?,
        col("z")?,
        col("y")?,
        col("x")?,
        col("radius")?,
        col("source")?,
    );
    let mut out = Vec::new();
    for line in lines {
        // names never contain commas in these exports, but be safe: split from the left
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < header.len() {
            continue;
        }
        let num = |i: usize| {
            fields[i]
                .parse::<f64>()
                .map_err(|e| Error::invalid(format!("{}: {e}", path.display())))
        };
        out.push(Annotation {
            t: num(ct)? as usize,
            z: num(cz)?,
            y: num(cy)?,
            x: num(cx)?,
            radius: num(cr)?,
            keyframe: fields[cs] == "keyframe",
        });
    }
    Ok(Some(out))
}

/// The calibration a movie gets; written by `dic_calibrate` as
/// `<image>/calibration/dic/summary.json`, read by `dic_maps`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub image: String,
    pub frames: Vec<usize>,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub pixel_um: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub z_step_um: f64,
    /// Bright-edge angle from the shell fit, degrees (image coords, x right, y down).
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_deg: f64,
    /// Spread of the per-frame shell θ (circular MAD), degrees.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_spread_deg: f64,
    /// Relief strength and symmetric term, relative to the medium level (plane median).
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub a: f64,
    /// The same in raw intensity units.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b_raw: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub a_raw: f64,
    /// Second-harmonic amplitude relative to the medium level (`cos 2φ` term).
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b2: f64,
    /// The two-harmonic fit `a + b·cos(φ − θ) + b2·cos 2(φ − θ2)`: its first
    /// harmonic (θ, b) and the second's phase θ2 (axial, degrees).
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_2h_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b_2h: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta2_deg: f64,
    /// Odd (radial-derivative) contrast fit: angle and strength (relative, per pixel).
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub odd_theta_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub odd_b: f64,
    /// Shell-free estimate: bright-edge angle from gradient moments, degrees.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_gradient_deg: f64,
    /// Orientation (mod 180) of the largest gradient energy, degrees.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub orientation_gradient_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub gradient_anisotropy: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub gradient_skewness: f64,
    /// `theta_deg − theta_gradient_deg`, wrapped, degrees.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub estimator_difference_deg: f64,
    /// Robust relief scale (1.4826·MAD of I − median, inside the shell), raw units.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub relief_scale: f64,
    /// Medium level (median of the planes), raw units.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub medium_level: f64,
    /// Eggshell ellipsoid, µm (median over the calibrated frames).
    pub shell: ShellRecord,
    /// Largest distance of a frame's shell centre from the median centre, µm.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub shell_drift_um: f64,
    /// The same in the image plane only (z of the centre is extrapolated when
    /// the stack is thinner than the egg).
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub shell_drift_xy_um: f64,
    pub per_frame: Vec<FrameRecord>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShellRecord {
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub cx: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub cy: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub cz: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub a: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub c: f64,
    /// Angle of the long (anterior–posterior) in-plane axis, degrees.
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub angle_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRecord {
    pub t: usize,
    pub shell: Option<ShellRecord>,
    pub planes_fitted: usize,
    pub planes_used: usize,
    /// Whether the frame entered the movie-level fit (a shell with relief
    /// at least `--min-relative-b` of the frames' median b).
    #[serde(default)]
    pub accepted: bool,
    pub samples: usize,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_se_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub a: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub b2: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_2h_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta2_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub residual: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub odd_theta_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub odd_b: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub theta_gradient_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub orientation_gradient_deg: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub gradient_anisotropy: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub gradient_skewness: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub medium_level: f64,
    #[serde(default = "nan", deserialize_with = "nan_if_null")]
    pub relief_scale: f64,
}

/// JSON has no NaN: `serde_json` writes it as `null`; read `null` back as NaN.
fn nan_if_null<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

fn nan() -> f64 {
    f64::NAN
}

pub fn calibration_path(image: &Path) -> PathBuf {
    image.join("calibration").join("dic").join("summary.json")
}

pub fn read_calibration(image: &Path) -> Result<Calibration> {
    let path = calibration_path(image);
    let text = fs::read_to_string(&path)
        .map_err(|e| Error::invalid(format!("{}: {e}; run dic_calibrate first", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::invalid(format!("{}: {e}", path.display())))
}

pub fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Circular median-like centre of angles (radians): the circular mean, then
/// the median of the wrapped deviations added back. Returns (centre, MAD).
pub fn circular_centre(angles: &[f64]) -> (f64, f64) {
    use blockflow::ops::dic::{angle_difference, wrap_angle};
    let finite: Vec<f64> = angles.iter().copied().filter(|a| a.is_finite()).collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let (s, c) = finite
        .iter()
        .fold((0.0, 0.0), |(s, c), a| (s + a.sin(), c + a.cos()));
    let mean = s.atan2(c);
    let dev: Vec<f64> = finite.iter().map(|&a| angle_difference(a, mean)).collect();
    let centre = wrap_angle(mean + median(&dev));
    let mad = median(
        &finite
            .iter()
            .map(|&a| angle_difference(a, centre).abs())
            .collect::<Vec<_>>(),
    );
    (centre, mad)
}

/// Write a CSV: header row then rows, values formatted by the caller.
pub fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(Error::backend)?;
    }
    let mut text = header.join(",");
    text.push('\n');
    for row in rows {
        text.push_str(&row.join(","));
        text.push('\n');
    }
    fs::write(path, text).map_err(Error::backend)
}

/// Keyframe annotations grouped by frame.
pub fn keyframes_by_frame(annotations: &[Annotation]) -> BTreeMap<usize, Vec<Annotation>> {
    let mut map: BTreeMap<usize, Vec<Annotation>> = BTreeMap::new();
    for a in annotations.iter().filter(|a| a.keyframe) {
        map.entry(a.t).or_default().push(*a);
    }
    map
}

pub fn fmt(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "NaN".into()
    }
}
