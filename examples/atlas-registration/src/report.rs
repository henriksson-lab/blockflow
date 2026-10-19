// SPDX-License-Identifier: MIT
// Original work for this crate.

// A registration measurement report, as JSON.
//
// Two modes. With no fixture, a synthetic Gaussian pair with a known sub-voxel
// translation is registered and the transformed probe is compared with the
// known answer. With a fixture directory, a recorded elastix run is the
// reference: its fixed and moving `.npy` volumes are registered natively, and
// the native transform at a probe point is compared with the recorded
// `TransformParameters` files and transformix's `outputpoints.txt`.
//
// The fixture is checked before it is believed: the transform files must
// reproduce the recorded output point, the output must be for the probe, and
// the probe must actually move (a registration recorded with an identity
// transform would make agreement meaningless).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use blockflow::error::{Error, Result};
use blockflow::npy::{self, OrderPolicy};
use blockflow::voxels::Voxels;
use ndarray::Array3;
use serde::Serialize;
use serde_json::Value;

use crate::compute::{
    params_from_elastix_text, run_with_diagnostics, ElastixSettingStatus, FitConvergence, Metric,
    TransformModel, VolumeFitParams, ELASTIX_SETTING_SUPPORT,
};
use crate::{ElastixTransform, RegisterParams};

const SHAPE: [usize; 3] = [9, 7, 5];
const FIXED_CENTER: [f64; 3] = [3.2, 3.1, 2.0];
const MOVING_CENTER: [f64; 3] = [3.95, 2.75, 2.4];
const BLOCK: usize = 4;
const TOLERANCE: f64 = 0.08;
const OUTPUT_POINTS_TOLERANCE: f64 = 1.0e-5;
const MANIFEST: &str = "registration-manifest.json";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FixtureReport {
    pub kind: String,
    pub provenance: Option<String>,
    pub fixed_path: Option<String>,
    pub fixed_bytes: Option<u64>,
    pub fixed_fnv1a64: Option<String>,
    pub moving_path: Option<String>,
    pub moving_bytes: Option<u64>,
    pub moving_fnv1a64: Option<String>,
    pub output_points_path: Option<String>,
    pub output_points_bytes: Option<u64>,
    pub output_points_fnv1a64: Option<String>,
    pub log_path: Option<String>,
    pub log_bytes: Option<u64>,
    pub log_fnv1a64: Option<String>,
    pub parameter_files: Vec<String>,
    pub parameter_file_bytes: Vec<u64>,
    pub parameter_file_fnv1a64: Vec<String>,
    pub command_line: Option<String>,
    pub elastix_version: Option<String>,
    pub transformix_version: Option<String>,
    pub fixed_shape: [usize; 3],
    pub moving_shape: [usize; 3],
    pub fixed_center: [f64; 3],
    pub moving_center: [f64; 3],
    pub expected_translation: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NativeReport {
    pub implementation: String,
    pub command: String,
    pub model: String,
    pub metric: String,
    pub block: usize,
    pub wall_seconds: f64,
    pub cpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub optimizer_substages: usize,
    pub optimizer_converged_by: String,
    pub optimizer_final_cost: f64,
    pub optimizer_final_step_norm: f64,
    pub optimizer_final_cost_delta: Option<f64>,
    /// The displacement of the probe, which is the translation for a
    /// translation model.
    pub transform_parameters: Vec<f64>,
    pub transformed_point: [f64; 3],
}

/// The recorded elastix run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReferenceReport {
    pub implementation: String,
    pub command: String,
    pub elastix_version: Option<String>,
    pub transformix_version: Option<String>,
    pub wall_seconds: Option<f64>,
    pub cpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub transform_files: Vec<String>,
    pub transform_file_bytes: Vec<u64>,
    pub transform_file_fnv1a64: Vec<String>,
    pub transformed_point: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AgreementReport {
    pub reference: String,
    pub max_abs_error: f64,
    pub tolerance: f64,
    pub within_tolerance: bool,
    pub downstream_atlas_coordinate_max_abs_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnsupportedDifference {
    pub setting: String,
    pub native: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegistrationReport {
    pub schema: u32,
    pub command: String,
    pub fixture: FixtureReport,
    pub reference: Option<ReferenceReport>,
    pub native: NativeReport,
    pub agreement: AgreementReport,
    pub unsupported_differences: Vec<UnsupportedDifference>,
    pub measurement_gap: Option<String>,
}

/// The report as pretty-printed JSON.
pub fn render_json(report: &RegistrationReport) -> String {
    let mut text = serde_json::to_string_pretty(report).expect("a report serialises");
    text.push('\n');
    text
}

/// Fail unless the native transform agrees with the reference within the
/// stated tolerance, at the probe and, if the fixture gives one, downstream.
pub fn require_threshold(report: &RegistrationReport) -> Result<()> {
    let agreement = &report.agreement;
    if !agreement.within_tolerance {
        return Err(Error::InvalidArgument(format!(
            "registration report: native max_abs_error {} exceeds tolerance {}",
            agreement.max_abs_error, agreement.tolerance
        )));
    }
    if let Some(downstream) = agreement.downstream_atlas_coordinate_max_abs_error {
        if downstream > agreement.tolerance {
            return Err(Error::InvalidArgument(format!(
                "registration report: downstream atlas coordinate max_abs_error {downstream} \
                 exceeds tolerance {}",
                agreement.tolerance
            )));
        }
    }
    Ok(())
}

// ------------------------------------------------------------- synthetic --

pub fn synthetic_translation_report() -> Result<RegistrationReport> {
    let fixed = gaussian(FIXED_CENTER);
    let moving = gaussian(MOVING_CENTER);
    let params = VolumeFitParams::default().with_metric(Metric::MeanSquares);
    let (native, _) = native_report(&params, &fixed, &moving, BLOCK, FIXED_CENTER)?;
    let max_abs_error = max_abs_error(native.transformed_point, MOVING_CENTER);
    Ok(RegistrationReport {
        schema: 1,
        command: "register-report".to_string(),
        fixture: FixtureReport {
            kind: "synthetic_translation_gaussian".to_string(),
            provenance: None,
            fixed_path: None,
            fixed_bytes: None,
            fixed_fnv1a64: None,
            moving_path: None,
            moving_bytes: None,
            moving_fnv1a64: None,
            output_points_path: None,
            output_points_bytes: None,
            output_points_fnv1a64: None,
            log_path: None,
            log_bytes: None,
            log_fnv1a64: None,
            parameter_files: Vec::new(),
            parameter_file_bytes: Vec::new(),
            parameter_file_fnv1a64: Vec::new(),
            command_line: None,
            elastix_version: None,
            transformix_version: None,
            fixed_shape: SHAPE,
            moving_shape: SHAPE,
            fixed_center: FIXED_CENTER,
            moving_center: MOVING_CENTER,
            expected_translation: std::array::from_fn(|a| MOVING_CENTER[a] - FIXED_CENTER[a]),
        },
        reference: None,
        native,
        agreement: AgreementReport {
            reference: "synthetic known translation".to_string(),
            max_abs_error,
            tolerance: TOLERANCE,
            within_tolerance: max_abs_error <= TOLERANCE,
            downstream_atlas_coordinate_max_abs_error: None,
        },
        unsupported_differences: unsupported_differences(),
        measurement_gap: None,
    })
}

fn gaussian(center: [f64; 3]) -> Voxels {
    Array3::from_shape_fn((SHAPE[0], SHAPE[1], SHAPE[2]), |(x, y, z)| {
        let dx = x as f64 - center[0];
        let dy = y as f64 - center[1];
        let dz = z as f64 - center[2];
        (-(dx * dx + dy * dy + dz * dz) / 3.0).exp()
    })
    .into()
}

fn native_report(
    params: &VolumeFitParams,
    fixed: &Voxels,
    moving: &Voxels,
    block: usize,
    probe: [f64; 3],
) -> Result<(NativeReport, ElastixTransform)> {
    let cpu_start = CpuSnapshot::now();
    let start = Instant::now();
    let diagnostics = run_with_diagnostics(params, fixed, moving, block, 1)?;
    let wall_seconds = start.elapsed().as_secs_f64();
    let cpu_seconds = cpu_start.and_then(CpuSnapshot::elapsed);
    let transformed_point = diagnostics
        .transform
        .apply(probe, &RegisterParams::default());
    let report = NativeReport {
        implementation: "blockflow-atlas-registration".to_string(),
        command: "blockflow_atlas_registration::compute::run_with_diagnostics".to_string(),
        model: model_name(params.model).to_string(),
        metric: metric_name(params.metric).to_string(),
        block,
        wall_seconds,
        cpu_seconds,
        peak_rss_bytes: peak_rss_bytes(),
        optimizer_substages: diagnostics.substages,
        optimizer_converged_by: convergence_name(diagnostics.converged_by).to_string(),
        optimizer_final_cost: diagnostics.final_cost,
        optimizer_final_step_norm: diagnostics.final_step_norm,
        optimizer_final_cost_delta: diagnostics.final_cost_delta,
        transform_parameters: (0..3).map(|a| transformed_point[a] - probe[a]).collect(),
        transformed_point,
    };
    Ok((report, diagnostics.transform))
}

fn convergence_name(convergence: FitConvergence) -> &'static str {
    match convergence {
        FitConvergence::StepTolerance => "step_tolerance",
        FitConvergence::CostTolerance => "cost_tolerance",
    }
}

fn model_name(model: TransformModel) -> &'static str {
    match model {
        TransformModel::Translation => "translation",
        TransformModel::Affine => "affine",
        TransformModel::BSpline => "bspline",
    }
}

fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::MeanSquares => "mean_squares",
        Metric::NormalizedCorrelation => "normalized_correlation",
        Metric::MutualInformation => "mutual_information",
    }
}

fn unsupported_differences() -> Vec<UnsupportedDifference> {
    ELASTIX_SETTING_SUPPORT
        .iter()
        .filter(|row| row.status == ElastixSettingStatus::Unsupported)
        .map(|row| UnsupportedDifference {
            setting: row.setting.to_string(),
            native: row.native.to_string(),
        })
        .collect()
}

// --------------------------------------------------------------- fixture --

/// Compare the native registration with a recorded elastix run in `root`.
///
/// `root/registration-manifest.json` names the files, relative to `root`:
///
/// ```json
/// {
///   "provenance": { "kind": "public" | "local-only" | "generated" },
///   "paths": { "fixed_image": "…npy", "moving_image": "…npy",
///              "output_points": "outputpoints.txt", "log": "elastix.log" },
///   "versions": { "elastix": "…", "transformix": "…" },
///   "command_line": "…",
///   "parameter_files": ["align_affine.txt"],
///   "transform_files": ["TransformParameters.0.txt"],
///   "non_identity_probe": { "input_point": [x, y, z], "minimum_abs_displacement": 0.1 },
///   "downstream_atlas_coordinate": { "input_point": […], "reference_point": […] },
///   "reference_runtime": { "wall_seconds": …, "cpu_seconds": …, "peak_rss_bytes": … },
///   "agreement": { "tolerance": 0.5 }
/// }
/// ```
///
/// `downstream_atlas_coordinate` and `reference_runtime` are optional; the
/// former may give a precomputed `native_point` instead of an `input_point`.
pub fn fixture_report(root: &Path) -> Result<RegistrationReport> {
    let manifest = Manifest::read(root)?;
    let fixed_path = fixture_file(root, &manifest.fixed_image, "paths.fixed_image")?;
    let moving_path = fixture_file(root, &manifest.moving_image, "paths.moving_image")?;
    let output_points_path = fixture_file(root, &manifest.output_points, "paths.output_points")?;
    let log_path = fixture_file(root, &manifest.log, "paths.log")?;
    let parameter_paths = manifest.parameter_paths(root)?;
    let parameter_texts = parameter_paths
        .iter()
        .map(|path| read_text(path, "parameter file"))
        .collect::<Result<Vec<_>>>()?;
    let fixed = read_f64_volume(&fixed_path, "fixed_image")?;
    let moving = read_f64_volume(&moving_path, "moving_image")?;
    let (params, parameter_gap) = manifest.native_params(&parameter_texts)?;
    let block = fixed.shape().iter().copied().max().unwrap_or(1);
    let (native, transform) = native_report(&params, &fixed, &moving, block, manifest.probe)?;
    let reference = manifest.reference_report(root, &output_points_path)?;
    let downstream = manifest.downstream_error(&transform)?;
    let max_abs_error = max_abs_error(native.transformed_point, reference.transformed_point);

    let mut gaps = Vec::new();
    if reference.wall_seconds.is_none() {
        gaps.push(
            "reference runtime/RSS are not recorded in the fixture; only the recorded transform \
             output is compared"
                .to_string(),
        );
    }
    gaps.extend(parameter_gap);
    let measured = |path: &Path| -> Result<(Option<u64>, Option<String>)> {
        Ok((Some(file_bytes(path)?), Some(file_fnv1a64(path)?)))
    };
    let (fixed_bytes, fixed_fnv1a64) = measured(&fixed_path)?;
    let (moving_bytes, moving_fnv1a64) = measured(&moving_path)?;
    let (output_points_bytes, output_points_fnv1a64) = measured(&output_points_path)?;
    let (log_bytes, log_fnv1a64) = measured(&log_path)?;
    Ok(RegistrationReport {
        schema: 1,
        command: "register-report".to_string(),
        fixture: FixtureReport {
            kind: "elastix_fixture".to_string(),
            provenance: Some(manifest.provenance.clone()),
            fixed_path: Some(manifest.fixed_image.clone()),
            fixed_bytes,
            fixed_fnv1a64,
            moving_path: Some(manifest.moving_image.clone()),
            moving_bytes,
            moving_fnv1a64,
            output_points_path: Some(manifest.output_points.clone()),
            output_points_bytes,
            output_points_fnv1a64,
            log_path: Some(manifest.log.clone()),
            log_bytes,
            log_fnv1a64,
            parameter_files: manifest.parameter_files.clone(),
            parameter_file_bytes: parameter_paths
                .iter()
                .map(|path| file_bytes(path))
                .collect::<Result<_>>()?,
            parameter_file_fnv1a64: parameter_paths
                .iter()
                .map(|path| file_fnv1a64(path))
                .collect::<Result<_>>()?,
            command_line: Some(manifest.command_line.clone()),
            elastix_version: Some(manifest.elastix_version.clone()),
            transformix_version: Some(manifest.transformix_version.clone()),
            fixed_shape: fixed.shape(),
            moving_shape: moving.shape(),
            fixed_center: manifest.probe,
            moving_center: reference.transformed_point,
            expected_translation: std::array::from_fn(|a| {
                reference.transformed_point[a] - manifest.probe[a]
            }),
        },
        reference: Some(reference),
        native,
        agreement: AgreementReport {
            reference: "recorded elastix transform files".to_string(),
            max_abs_error,
            tolerance: manifest.tolerance,
            within_tolerance: max_abs_error <= manifest.tolerance,
            downstream_atlas_coordinate_max_abs_error: downstream,
        },
        unsupported_differences: unsupported_differences(),
        measurement_gap: (!gaps.is_empty()).then(|| gaps.join("; ")),
    })
}

#[derive(Debug, Clone, PartialEq)]
struct Manifest {
    provenance: String,
    fixed_image: String,
    moving_image: String,
    output_points: String,
    log: String,
    command_line: String,
    elastix_version: String,
    transformix_version: String,
    parameter_files: Vec<String>,
    transform_files: Vec<String>,
    probe: [f64; 3],
    minimum_abs_displacement: f64,
    downstream_input_point: Option<[f64; 3]>,
    downstream_native_point: Option<[f64; 3]>,
    downstream_reference_point: Option<[f64; 3]>,
    tolerance: f64,
    wall_seconds: Option<f64>,
    cpu_seconds: Option<f64>,
    peak_rss_bytes: Option<u64>,
}

impl Manifest {
    fn read(root: &Path) -> Result<Self> {
        let path = root.join(MANIFEST);
        let text = fs::read_to_string(&path).map_err(|error| {
            Error::InvalidArgument(format!(
                "{}: registration fixture manifest could not be read: {error}",
                path.display()
            ))
        })?;
        let json: Value = serde_json::from_str(&text).map_err(|error| {
            Error::InvalidArgument(format!(
                "{}: registration fixture manifest is not JSON: {error}",
                path.display()
            ))
        })?;
        let provenance = string(&json, "provenance.kind")?;
        if !matches!(provenance.as_str(), "public" | "local-only" | "generated") {
            return Err(Error::InvalidArgument(format!(
                "registration fixture manifest: `provenance.kind` must be public, local-only, or \
                 generated, got `{provenance}`"
            )));
        }
        let minimum_abs_displacement = positive(
            number(&json, "non_identity_probe.minimum_abs_displacement")?,
            "non_identity_probe.minimum_abs_displacement",
        )?;
        let tolerance = match optional_number(&json, "agreement.tolerance")? {
            Some(value) => positive(value, "agreement.tolerance")?,
            None => 0.5,
        };
        let non_negative_runtime = |key: &str| -> Result<Option<f64>> {
            let field = format!("reference_runtime.{key}");
            optional_number(&json, &field)?
                .map(|value| non_negative(value, &field))
                .transpose()
        };
        let peak_rss_bytes = optional_number(&json, "reference_runtime.peak_rss_bytes")?
            .map(|value| positive_u64(value, "reference_runtime.peak_rss_bytes"))
            .transpose()?;
        Ok(Self {
            provenance,
            fixed_image: string(&json, "paths.fixed_image")?,
            moving_image: string(&json, "paths.moving_image")?,
            output_points: string(&json, "paths.output_points")?,
            log: string(&json, "paths.log")?,
            command_line: string(&json, "command_line")?,
            elastix_version: string(&json, "versions.elastix")?,
            transformix_version: string(&json, "versions.transformix")?,
            parameter_files: strings(&json, "parameter_files")?,
            transform_files: strings(&json, "transform_files")?,
            probe: point(&json, "non_identity_probe.input_point")?,
            minimum_abs_displacement,
            downstream_input_point: optional_point(
                &json,
                "downstream_atlas_coordinate.input_point",
            )?,
            downstream_native_point: optional_point(
                &json,
                "downstream_atlas_coordinate.native_point",
            )?,
            downstream_reference_point: optional_point(
                &json,
                "downstream_atlas_coordinate.reference_point",
            )?,
            tolerance,
            wall_seconds: non_negative_runtime("wall_seconds")?,
            cpu_seconds: non_negative_runtime("cpu_seconds")?,
            peak_rss_bytes,
        })
    }

    fn reference_report(&self, root: &Path, output_points: &Path) -> Result<ReferenceReport> {
        if self.transform_files.is_empty() {
            return Err(Error::InvalidArgument(
                "registration fixture manifest: `transform_files` must name at least one file"
                    .to_string(),
            ));
        }
        let paths = self
            .transform_files
            .iter()
            .map(|relative| fixture_file(root, relative, "transform_files"))
            .collect::<Result<Vec<_>>>()?;
        let texts = paths
            .iter()
            .map(|path| read_text(path, "transform file"))
            .collect::<Result<Vec<_>>>()?;
        let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let transform = ElastixTransform::from_texts(&refs)?;
        let from_files = transform.apply(self.probe, &RegisterParams::default());
        let (input, transformed_point) = read_output_point(output_points)?;
        let input_error = max_abs_error(input, self.probe);
        if input_error > OUTPUT_POINTS_TOLERANCE {
            return Err(Error::InvalidArgument(format!(
                "registration fixture paths.output_points InputPoint {input:?} does not match \
                 non_identity_probe.input_point {:?}: max_abs_error {input_error} exceeds \
                 {OUTPUT_POINTS_TOLERANCE}",
                self.probe
            )));
        }
        let output_error = max_abs_error(from_files, transformed_point);
        if output_error > OUTPUT_POINTS_TOLERANCE {
            return Err(Error::InvalidArgument(format!(
                "registration fixture paths.output_points disagrees with transform_files at probe \
                 {:?}: max_abs_error {output_error} exceeds {OUTPUT_POINTS_TOLERANCE}",
                self.probe
            )));
        }
        let displacement = max_abs_error(transformed_point, self.probe);
        if displacement < self.minimum_abs_displacement {
            return Err(Error::InvalidArgument(format!(
                "registration fixture transform displacement {displacement} is below manifest \
                 minimum {}",
                self.minimum_abs_displacement
            )));
        }
        Ok(ReferenceReport {
            implementation: "elastix fixture".to_string(),
            command: self.command_line.clone(),
            elastix_version: Some(self.elastix_version.clone()),
            transformix_version: Some(self.transformix_version.clone()),
            wall_seconds: self.wall_seconds,
            cpu_seconds: self.cpu_seconds,
            peak_rss_bytes: self.peak_rss_bytes,
            transform_files: self.transform_files.clone(),
            transform_file_bytes: paths
                .iter()
                .map(|path| file_bytes(path))
                .collect::<Result<_>>()?,
            transform_file_fnv1a64: paths
                .iter()
                .map(|path| file_fnv1a64(path))
                .collect::<Result<_>>()?,
            transformed_point,
        })
    }

    fn parameter_paths(&self, root: &Path) -> Result<Vec<PathBuf>> {
        if self.parameter_files.is_empty() {
            return Err(Error::InvalidArgument(
                "registration fixture manifest: `parameter_files` must name at least one file"
                    .to_string(),
            ));
        }
        self.parameter_files
            .iter()
            .map(|relative| fixture_file(root, relative, "parameter_files"))
            .collect()
    }

    /// The downstream error, computing the native point from the input point
    /// when one is given.
    fn downstream_error(&self, native_transform: &ElastixTransform) -> Result<Option<f64>> {
        match (
            self.downstream_input_point,
            self.downstream_native_point,
            self.downstream_reference_point,
        ) {
            (Some(input), None, Some(reference)) => {
                let native = native_transform.apply(input, &RegisterParams::default());
                Ok(Some(max_abs_error(native, reference)))
            }
            (None, Some(native), Some(reference)) => Ok(Some(max_abs_error(native, reference))),
            (None, None, None) => Ok(None),
            (Some(_), Some(_), Some(_)) => Err(Error::InvalidArgument(
                "registration fixture manifest: `downstream_atlas_coordinate` must provide either \
                 `input_point` plus `reference_point`, or a precomputed `native_point` plus \
                 `reference_point`, not both native forms"
                    .to_string(),
            )),
            _ => Err(Error::InvalidArgument(
                "registration fixture manifest: `downstream_atlas_coordinate` must provide \
                 `input_point` plus `reference_point`, a precomputed `native_point` plus \
                 `reference_point`, or neither"
                    .to_string(),
            )),
        }
    }

    /// One parameter file maps to one native configuration. Several (an affine
    /// then a B-spline, say) do not yet: the default translation is measured
    /// instead and the gap says so.
    fn native_params(&self, texts: &[String]) -> Result<(VolumeFitParams, Option<String>)> {
        match texts {
            [text] => {
                let name = &self.parameter_files[0];
                params_from_elastix_text(text)
                    .map(|params| (params, None))
                    .map_err(|error| {
                        Error::InvalidArgument(format!(
                            "registration fixture manifest `parameter_files` entry `{name}` does \
                             not map to a native registration: {error}"
                        ))
                    })
            }
            many => Ok((
                VolumeFitParams::default().with_metric(Metric::MeanSquares),
                Some(format!(
                    "the report cannot yet derive one native configuration from {} elastix \
                     parameter files; the default translation was measured instead",
                    many.len()
                )),
            )),
        }
    }
}

// ----------------------------------------------------------- the manifest --

fn lookup<'a>(json: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(json, |at, key| at.get(key))
        .filter(|value| !value.is_null())
}

fn required<'a>(json: &'a Value, path: &str) -> Result<&'a Value> {
    lookup(json, path).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "registration fixture manifest: `{path}` is missing"
        ))
    })
}

fn string(json: &Value, path: &str) -> Result<String> {
    required(json, path)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "registration fixture manifest: `{path}` must be a string"
            ))
        })
}

fn strings(json: &Value, path: &str) -> Result<Vec<String>> {
    let values = required(json, path)?.as_array().ok_or_else(|| {
        Error::InvalidArgument(format!(
            "registration fixture manifest: `{path}` must be an array of strings"
        ))
    })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "registration fixture manifest: `{path}[{index}]` must be a string"
                ))
            })
        })
        .collect()
}

fn number(json: &Value, path: &str) -> Result<f64> {
    optional_number(json, path)?.ok_or_else(|| {
        Error::InvalidArgument(format!(
            "registration fixture manifest: `{path}` is missing"
        ))
    })
}

fn optional_number(json: &Value, path: &str) -> Result<Option<f64>> {
    lookup(json, path)
        .map(|value| {
            value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "registration fixture manifest: `{path}` must be a finite number"
                ))
            })
        })
        .transpose()
}

fn point(json: &Value, path: &str) -> Result<[f64; 3]> {
    optional_point(json, path)?.ok_or_else(|| {
        Error::InvalidArgument(format!(
            "registration fixture manifest: `{path}` is missing"
        ))
    })
}

fn optional_point(json: &Value, path: &str) -> Result<Option<[f64; 3]>> {
    let Some(value) = lookup(json, path) else {
        return Ok(None);
    };
    let wrong = || {
        Error::InvalidArgument(format!(
            "registration fixture manifest: `{path}` must be three finite numbers"
        ))
    };
    let values = value.as_array().ok_or_else(wrong)?;
    if values.len() != 3 {
        return Err(wrong());
    }
    let mut out = [0.0; 3];
    for axis in 0..3 {
        out[axis] = values[axis]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(wrong)?;
    }
    Ok(Some(out))
}

fn positive(value: f64, field: &str) -> Result<f64> {
    if !(value > 0.0) {
        return Err(Error::InvalidArgument(format!(
            "registration fixture manifest: `{field}` must be positive, got {value}"
        )));
    }
    Ok(value)
}

fn non_negative(value: f64, field: &str) -> Result<f64> {
    if !(value >= 0.0) {
        return Err(Error::InvalidArgument(format!(
            "registration fixture manifest: `{field}` must be non-negative, got {value}"
        )));
    }
    Ok(value)
}

fn positive_u64(value: f64, field: &str) -> Result<u64> {
    if !(value > 0.0) || value.fract() != 0.0 || value > u64::MAX as f64 {
        return Err(Error::InvalidArgument(format!(
            "registration fixture manifest: `{field}` must be a positive integer byte count, got \
             {value}"
        )));
    }
    Ok(value as u64)
}

// ---------------------------------------------------------------- files --

fn fixture_file(root: &Path, relative: &str, field: &str) -> Result<PathBuf> {
    if Path::new(relative).is_absolute() {
        return Err(Error::InvalidArgument(format!(
            "registration fixture manifest `{field}` must be relative to the fixture root, got \
             `{relative}`"
        )));
    }
    let path = root.join(relative);
    if !path.is_file() {
        return Err(Error::InvalidArgument(format!(
            "registration fixture manifest `{field}` points to missing file `{}`",
            path.display()
        )));
    }
    Ok(path)
}

fn read_text(path: &Path, what: &str) -> Result<String> {
    fs::read_to_string(path).map_err(|error| {
        Error::InvalidArgument(format!(
            "{}: registration fixture {what} could not be read: {error}",
            path.display()
        ))
    })
}

fn read_f64_volume(path: &Path, field: &str) -> Result<Voxels> {
    let voxels = npy::read_voxels_file(path, OrderPolicy::Either)?;
    if voxels.dtype() != blockflow::Dtype::F64 {
        return Err(Error::InvalidArgument(format!(
            "registration fixture manifest `{field}` points to `{}`, whose dtype is {:?}; the \
             report needs f64 .npy inputs so both registrations read the same values",
            path.display(),
            voxels.dtype()
        )));
    }
    Ok(voxels)
}

/// The first row's `InputPoint` and `OutputPoint` from transformix's
/// `outputpoints.txt`.
fn read_output_point(path: &Path) -> Result<([f64; 3], [f64; 3])> {
    let text = read_text(path, "output points")?;
    for line in text.lines() {
        let Some(input) = output_column(path, line, "InputPoint")? else {
            continue;
        };
        let output = output_column(path, line, "OutputPoint")?.ok_or_else(|| {
            Error::InvalidArgument(format!(
                "registration fixture paths.output_points `{}` row with InputPoint has no \
                 OutputPoint column",
                path.display()
            ))
        })?;
        return Ok((input, output));
    }
    Err(Error::InvalidArgument(format!(
        "registration fixture paths.output_points `{}` has no InputPoint column",
        path.display()
    )))
}

fn output_column(path: &Path, line: &str, name: &str) -> Result<Option<[f64; 3]>> {
    let column = format!("{name} = [");
    let Some(start) = line.find(&column) else {
        return Ok(None);
    };
    let rest = &line[start + column.len()..];
    let bad = |what: String| {
        Error::InvalidArgument(format!(
            "registration fixture paths.output_points `{}` {what}",
            path.display()
        ))
    };
    let end = rest
        .find(']')
        .ok_or_else(|| bad(format!("has an unterminated {name} column")))?;
    let values = rest[..end]
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|error| bad(format!("contains `{token}` in {name}: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;
    match values.as_slice() {
        [a, b, c] => Ok(Some([*a, *b, *c])),
        other => Err(bad(format!(
            "{name} must contain three numbers, got {}",
            other.len()
        ))),
    }
}

fn file_bytes(path: &Path) -> Result<u64> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            Error::InvalidArgument(format!(
                "{}: registration fixture file metadata could not be read: {error}",
                path.display()
            ))
        })
}

/// FNV-1a, 64 bit: a cheap fingerprint so a report says which bytes it read.
fn file_fnv1a64(path: &Path) -> Result<String> {
    let fail = |error: std::io::Error| {
        Error::InvalidArgument(format!(
            "{}: registration fixture file could not be fingerprinted: {error}",
            path.display()
        ))
    };
    let mut file = fs::File::open(path).map_err(fail)?;
    let mut hash = 0xcbf29ce484222325_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let read = file.read(&mut buffer).map_err(fail)?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn max_abs_error(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|a| (left[a] - right[a]).abs())
        .fold(0.0, f64::max)
}

// -------------------------------------------------------------- process --

fn peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let kb = status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?;
        value.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    Some(kb * 1024)
}

#[derive(Debug, Clone, Copy)]
struct CpuSnapshot {
    ticks: u64,
    ticks_per_second: u64,
}

impl CpuSnapshot {
    fn now() -> Option<Self> {
        Some(Self {
            ticks: process_cpu_ticks()?,
            ticks_per_second: ticks_per_second()?,
        })
    }

    fn elapsed(self) -> Option<f64> {
        let end = process_cpu_ticks()?;
        Some(end.saturating_sub(self.ticks) as f64 / self.ticks_per_second as f64)
    }
}

fn process_cpu_ticks() -> Option<u64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    let after_name = stat.rsplit_once(") ")?.1;
    let fields = after_name.split_whitespace().collect::<Vec<_>>();
    let user = fields.get(11)?.parse::<u64>().ok()?;
    let system = fields.get(12)?.parse::<u64>().ok()?;
    Some(user + system)
}

fn ticks_per_second() -> Option<u64> {
    let output = Command::new("getconf").arg("CLK_TCK").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|ticks| *ticks > 0)
}
