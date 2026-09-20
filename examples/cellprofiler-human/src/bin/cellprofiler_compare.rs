// SPDX-License-Identifier: MIT

use std::fs;
use std::path::{Path, PathBuf};

use blockflow::{compare_labels, Error, Result};
use clap::Parser;
use ndarray::Array3;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "cellprofiler-compare")]
struct Cli {
    #[arg(long)]
    blockflow: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    blockflow_labels: Option<PathBuf>,
    #[arg(long)]
    reference_labels: Option<PathBuf>,
    #[arg(long, default_value = "cellprofiler-comparison.json")]
    out: PathBuf,
    #[arg(long, default_value_t = 5.0)]
    max_centroid_distance: f64,
    #[arg(long, default_value_t = 0.20)]
    max_area_relative_error: f64,
    #[arg(long, default_value_t = 0.20)]
    max_mean_intensity_relative_error: f64,
    #[arg(long, default_value_t = 0.5)]
    min_label_overlap: f64,
    #[arg(long, default_value_t = 0.5)]
    min_mean_label_iou: f64,
    #[arg(long)]
    blockflow_area: Option<String>,
    #[arg(long)]
    blockflow_centroid_z: Option<String>,
    #[arg(long)]
    blockflow_centroid_y: Option<String>,
    #[arg(long)]
    blockflow_centroid_x: Option<String>,
    #[arg(long)]
    blockflow_mean_intensity: Option<String>,
    #[arg(long)]
    blockflow_integrated_intensity: Option<String>,
    #[arg(long)]
    blockflow_bbox_min_z: Option<String>,
    #[arg(long)]
    blockflow_bbox_min_y: Option<String>,
    #[arg(long)]
    blockflow_bbox_min_x: Option<String>,
    #[arg(long)]
    blockflow_bbox_max_z: Option<String>,
    #[arg(long)]
    blockflow_bbox_max_y: Option<String>,
    #[arg(long)]
    blockflow_bbox_max_x: Option<String>,
    #[arg(long)]
    reference_area: Option<String>,
    #[arg(long)]
    reference_centroid_z: Option<String>,
    #[arg(long)]
    reference_centroid_y: Option<String>,
    #[arg(long)]
    reference_centroid_x: Option<String>,
    #[arg(long)]
    reference_mean_intensity: Option<String>,
    #[arg(long)]
    reference_integrated_intensity: Option<String>,
    #[arg(long)]
    reference_bbox_min_z: Option<String>,
    #[arg(long)]
    reference_bbox_min_y: Option<String>,
    #[arg(long)]
    reference_bbox_min_x: Option<String>,
    #[arg(long)]
    reference_bbox_max_z: Option<String>,
    #[arg(long)]
    reference_bbox_max_y: Option<String>,
    #[arg(long)]
    reference_bbox_max_x: Option<String>,
}

#[derive(Debug)]
struct Config {
    blockflow: PathBuf,
    reference: PathBuf,
    blockflow_labels: Option<PathBuf>,
    reference_labels: Option<PathBuf>,
    out: PathBuf,
    max_centroid_distance: f64,
    max_area_relative_error: f64,
    max_mean_intensity_relative_error: f64,
    min_label_overlap: f64,
    min_mean_label_iou: f64,
    blockflow_columns: ColumnConfig,
    reference_columns: ColumnConfig,
}

impl Config {
    fn parse() -> Result<Self> {
        let cli = Cli::parse();
        let mut blockflow_columns = ColumnConfig::blockflow_defaults();
        let mut reference_columns = ColumnConfig::cellprofiler_defaults();

        if let Some(value) = cli.blockflow_area {
            blockflow_columns.area = Some(value);
        }
        if let Some(value) = cli.blockflow_centroid_z {
            blockflow_columns.centroid_z = Some(value);
        }
        if let Some(value) = cli.blockflow_centroid_y {
            blockflow_columns.centroid_y = Some(value);
        }
        if let Some(value) = cli.blockflow_centroid_x {
            blockflow_columns.centroid_x = Some(value);
        }
        if let Some(value) = cli.blockflow_mean_intensity {
            blockflow_columns.mean_intensity = Some(value);
        }
        if let Some(value) = cli.blockflow_integrated_intensity {
            blockflow_columns.integrated_intensity = Some(value);
        }
        if let Some(value) = cli.blockflow_bbox_min_z {
            blockflow_columns.bbox_min_z = Some(value);
        }
        if let Some(value) = cli.blockflow_bbox_min_y {
            blockflow_columns.bbox_min_y = Some(value);
        }
        if let Some(value) = cli.blockflow_bbox_min_x {
            blockflow_columns.bbox_min_x = Some(value);
        }
        if let Some(value) = cli.blockflow_bbox_max_z {
            blockflow_columns.bbox_max_z = Some(value);
        }
        if let Some(value) = cli.blockflow_bbox_max_y {
            blockflow_columns.bbox_max_y = Some(value);
        }
        if let Some(value) = cli.blockflow_bbox_max_x {
            blockflow_columns.bbox_max_x = Some(value);
        }
        if let Some(value) = cli.reference_area {
            reference_columns.area = Some(value);
        }
        if let Some(value) = cli.reference_centroid_z {
            reference_columns.centroid_z = Some(value);
        }
        if let Some(value) = cli.reference_centroid_y {
            reference_columns.centroid_y = Some(value);
        }
        if let Some(value) = cli.reference_centroid_x {
            reference_columns.centroid_x = Some(value);
        }
        if let Some(value) = cli.reference_mean_intensity {
            reference_columns.mean_intensity = Some(value);
        }
        if let Some(value) = cli.reference_integrated_intensity {
            reference_columns.integrated_intensity = Some(value);
        }
        if let Some(value) = cli.reference_bbox_min_z {
            reference_columns.bbox_min_z = Some(value);
        }
        if let Some(value) = cli.reference_bbox_min_y {
            reference_columns.bbox_min_y = Some(value);
        }
        if let Some(value) = cli.reference_bbox_min_x {
            reference_columns.bbox_min_x = Some(value);
        }
        if let Some(value) = cli.reference_bbox_max_z {
            reference_columns.bbox_max_z = Some(value);
        }
        if let Some(value) = cli.reference_bbox_max_y {
            reference_columns.bbox_max_y = Some(value);
        }
        if let Some(value) = cli.reference_bbox_max_x {
            reference_columns.bbox_max_x = Some(value);
        }

        Ok(Self {
            blockflow: cli.blockflow,
            reference: cli.reference,
            blockflow_labels: cli.blockflow_labels,
            reference_labels: cli.reference_labels,
            out: cli.out,
            max_centroid_distance: cli.max_centroid_distance,
            max_area_relative_error: cli.max_area_relative_error,
            max_mean_intensity_relative_error: cli.max_mean_intensity_relative_error,
            min_label_overlap: cli.min_label_overlap,
            min_mean_label_iou: cli.min_mean_label_iou,
            blockflow_columns,
            reference_columns,
        })
    }
}

#[derive(Debug, Clone)]
struct ColumnConfig {
    area: Option<String>,
    centroid_z: Option<String>,
    centroid_y: Option<String>,
    centroid_x: Option<String>,
    mean_intensity: Option<String>,
    integrated_intensity: Option<String>,
    bbox_min_z: Option<String>,
    bbox_min_y: Option<String>,
    bbox_min_x: Option<String>,
    bbox_max_z: Option<String>,
    bbox_max_y: Option<String>,
    bbox_max_x: Option<String>,
}

impl ColumnConfig {
    fn blockflow_defaults() -> Self {
        Self {
            area: Some("count".to_string()),
            centroid_z: Some("centroid_z".to_string()),
            centroid_y: Some("centroid_y".to_string()),
            centroid_x: Some("centroid_x".to_string()),
            mean_intensity: Some("intensity_mean".to_string()),
            integrated_intensity: Some("intensity_sum".to_string()),
            bbox_min_z: Some("bbox_min_z".to_string()),
            bbox_min_y: Some("bbox_min_y".to_string()),
            bbox_min_x: Some("bbox_min_x".to_string()),
            bbox_max_z: Some("bbox_max_z".to_string()),
            bbox_max_y: Some("bbox_max_y".to_string()),
            bbox_max_x: Some("bbox_max_x".to_string()),
        }
    }

    fn cellprofiler_defaults() -> Self {
        Self {
            area: Some("AreaShape_Area".to_string()),
            centroid_z: None,
            centroid_y: Some("Location_Center_Y".to_string()),
            centroid_x: Some("Location_Center_X".to_string()),
            mean_intensity: Some("Intensity_MeanIntensity_DNA".to_string()),
            integrated_intensity: Some("Intensity_IntegratedIntensity_DNA".to_string()),
            bbox_min_z: None,
            bbox_min_y: Some("AreaShape_BoundingBoxMinimum_Y".to_string()),
            bbox_min_x: Some("AreaShape_BoundingBoxMinimum_X".to_string()),
            bbox_max_z: None,
            bbox_max_y: Some("AreaShape_BoundingBoxMaximum_Y".to_string()),
            bbox_max_x: Some("AreaShape_BoundingBoxMaximum_X".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
struct ObjectRow {
    index: usize,
    area: Option<f64>,
    centroid: [f64; 3],
    mean_intensity: Option<f64>,
    integrated_intensity: Option<f64>,
    bbox_min: Option<[f64; 3]>,
    bbox_max: Option<[f64; 3]>,
}

#[derive(Debug)]
struct MatchedPair {
    blockflow: usize,
    reference: usize,
    centroid_distance: f64,
    area_relative_error: Option<f64>,
    mean_intensity_relative_error: Option<f64>,
    integrated_intensity_relative_error: Option<f64>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    let blockflow = read_objects(&config.blockflow, &config.blockflow_columns)?;
    let reference = read_objects(&config.reference, &config.reference_columns)?;
    let pairs = match_by_centroid(&blockflow, &reference, Some(config.max_centroid_distance));
    let matched_reference = pairs.iter().map(|pair| pair.reference).collect::<Vec<_>>();
    let unmatched_reference = (0..reference.len())
        .filter(|index| !matched_reference.contains(index))
        .collect::<Vec<_>>();
    let unmatched_blockflow = (0..blockflow.len())
        .filter(|index| !pairs.iter().any(|pair| pair.blockflow == *index))
        .collect::<Vec<_>>();

    let max_centroid_distance = pairs
        .iter()
        .map(|pair| pair.centroid_distance)
        .fold(0.0, f64::max);
    let mean_centroid_distance = mean(pairs.iter().map(|pair| pair.centroid_distance));
    let mean_area_relative_error = mean(pairs.iter().filter_map(|pair| pair.area_relative_error));
    let max_area_relative_error = pairs
        .iter()
        .filter_map(|pair| pair.area_relative_error)
        .fold(0.0, f64::max);
    let mean_mean_intensity_relative_error = mean(
        pairs
            .iter()
            .filter_map(|pair| pair.mean_intensity_relative_error),
    );
    let max_mean_intensity_relative_error = pairs
        .iter()
        .filter_map(|pair| pair.mean_intensity_relative_error)
        .fold(0.0, f64::max);
    let label_report = label_report(&config)?;
    let label_pass = label_report.as_ref().is_none_or(|report| report.passed);
    let centroid_failures = matches_over_threshold(
        &pairs,
        |pair| Some(pair.centroid_distance),
        config.max_centroid_distance,
        10,
    );
    let area_failures = matches_over_threshold(
        &pairs,
        |pair| pair.area_relative_error,
        config.max_area_relative_error,
        10,
    );
    let mean_intensity_failures = matches_over_threshold(
        &pairs,
        |pair| pair.mean_intensity_relative_error,
        config.max_mean_intensity_relative_error,
        10,
    );

    let passed = unmatched_blockflow.is_empty()
        && unmatched_reference.is_empty()
        && max_centroid_distance <= config.max_centroid_distance
        && max_area_relative_error <= config.max_area_relative_error
        && max_mean_intensity_relative_error <= config.max_mean_intensity_relative_error
        && label_pass;

    let report = json!({
        "passed": passed,
        "blockflow_objects": blockflow.len(),
        "reference_objects": reference.len(),
        "matched_objects": pairs.len(),
        "unmatched_blockflow": unmatched_blockflow,
        "unmatched_reference": unmatched_reference,
        "matching": {
            "method": "greedy_centroid_distance",
            "max_centroid_distance": config.max_centroid_distance,
            "out_of_gate_objects_are_unmatched": true,
        },
        "thresholds": {
            "max_centroid_distance": config.max_centroid_distance,
            "max_area_relative_error": config.max_area_relative_error,
            "max_mean_intensity_relative_error": config.max_mean_intensity_relative_error,
            "min_label_overlap": config.min_label_overlap,
            "min_mean_label_iou": config.min_mean_label_iou,
        },
        "metrics": {
            "max_centroid_distance": max_centroid_distance,
            "mean_centroid_distance": mean_centroid_distance,
            "max_area_relative_error": max_area_relative_error,
            "mean_area_relative_error": mean_area_relative_error,
            "max_mean_intensity_relative_error": max_mean_intensity_relative_error,
            "mean_mean_intensity_relative_error": mean_mean_intensity_relative_error,
        },
        "failure_summary": {
            "unmatched_blockflow": unmatched_blockflow.len(),
            "unmatched_reference": unmatched_reference.len(),
            "centroid_threshold_failures": centroid_failures.count,
            "area_threshold_failures": area_failures.count,
            "mean_intensity_threshold_failures": mean_intensity_failures.count,
            "label_comparison_failed": !label_pass,
        },
        "diagnostics": {
            "worst_centroid_matches": worst_matches_by(&pairs, |pair| Some(pair.centroid_distance), 10),
            "worst_area_matches": worst_matches_by(&pairs, |pair| pair.area_relative_error, 10),
            "worst_mean_intensity_matches": worst_matches_by(&pairs, |pair| pair.mean_intensity_relative_error, 10),
            "centroid_threshold_failures": centroid_failures.examples,
            "area_threshold_failures": area_failures.examples,
            "mean_intensity_threshold_failures": mean_intensity_failures.examples,
            "unmatched_blockflow": unmatched_blockflow.iter().map(|&index| {
                object_with_nearest_summary(&blockflow[index], &reference, "nearest_reference", 3)
            }).collect::<Vec<_>>(),
            "unmatched_reference": unmatched_reference.iter().map(|&index| {
                object_with_nearest_summary(&reference[index], &blockflow, "nearest_blockflow", 3)
            }).collect::<Vec<_>>(),
            "possible_reference_splits": split_merge_candidates(
                &reference,
                &unmatched_reference,
                &blockflow,
                "reference_row",
                "blockflow_parts",
            ),
            "possible_blockflow_merges": split_merge_candidates(
                &blockflow,
                &unmatched_blockflow,
                &reference,
                "blockflow_row",
                "reference_parts",
            ),
        },
        "label_agreement": label_report.map(LabelReport::into_json),
        "matches": pairs.iter().map(|pair| {
            json!({
                "blockflow_row": pair.blockflow,
                "reference_row": pair.reference,
                "centroid_distance": pair.centroid_distance,
                "area_relative_error": pair.area_relative_error,
                "mean_intensity_relative_error": pair.mean_intensity_relative_error,
                "integrated_intensity_relative_error": pair.integrated_intensity_relative_error,
            })
        }).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&report)
        .map_err(|err| Error::invalid(format!("cellprofiler-compare: encode JSON: {err}")))?;
    fs::write(&config.out, format!("{text}\n")).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-compare: write {}: {err}",
            config.out.display()
        ))
    })?;
    println!(
        "passed={} matched={} blockflow={} reference={} output={}",
        passed,
        pairs.len(),
        blockflow.len(),
        reference.len(),
        config.out.display()
    );
    Ok(())
}

#[derive(Debug)]
struct LabelReport {
    passed: bool,
    foreground_dice: f64,
    foreground_jaccard: f64,
    mean_matched_iou: Option<f64>,
    truth_objects: usize,
    produced_objects: usize,
    matched: usize,
    split: usize,
    merged: usize,
    missed: usize,
    spurious: usize,
    min_overlap: f64,
}

impl LabelReport {
    fn into_json(self) -> serde_json::Value {
        json!({
            "passed": self.passed,
            "foreground_dice": self.foreground_dice,
            "foreground_jaccard": self.foreground_jaccard,
            "mean_matched_iou": self.mean_matched_iou,
            "truth_objects": self.truth_objects,
            "produced_objects": self.produced_objects,
            "matched": self.matched,
            "split": self.split,
            "merged": self.merged,
            "missed": self.missed,
            "spurious": self.spurious,
            "min_overlap": self.min_overlap,
        })
    }
}

fn label_report(config: &Config) -> Result<Option<LabelReport>> {
    match (&config.reference_labels, &config.blockflow_labels) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(Error::invalid(
            "cellprofiler-compare: label comparison needs both --reference-labels and \
             --blockflow-labels",
        )),
        (Some(reference), Some(blockflow)) => {
            let reference = load_label_image(reference)?;
            let blockflow = load_label_image(blockflow)?;
            let foreground = foreground_overlap(reference.view(), blockflow.view())?;
            let agreement =
                compare_labels(reference.view(), blockflow.view(), config.min_label_overlap)?;
            let mean_matched_iou = mean(agreement.matched.iter().map(|matched| matched.iou));
            let passed = agreement.is_exact()
                && mean_matched_iou.is_some_and(|iou| iou >= config.min_mean_label_iou);
            Ok(Some(LabelReport {
                passed,
                foreground_dice: foreground.dice,
                foreground_jaccard: foreground.jaccard,
                mean_matched_iou,
                truth_objects: agreement.truth_objects,
                produced_objects: agreement.produced_objects,
                matched: agreement.matched.len(),
                split: agreement.split.len(),
                merged: agreement.merged.len(),
                missed: agreement.missed.len(),
                spurious: agreement.spurious.len(),
                min_overlap: agreement.min_overlap,
            }))
        }
    }
}

fn load_label_image(path: &Path) -> Result<Array3<u32>> {
    let image = image::ImageReader::open(path)
        .map_err(|err| {
            Error::invalid(format!(
                "cellprofiler-compare: open label image {}: {err}",
                path.display()
            ))
        })?
        .decode()
        .map_err(|err| {
            Error::invalid(format!(
                "cellprofiler-compare: decode label image {}: {err}",
                path.display()
            ))
        })?
        .to_luma16();
    let (width, height) = image.dimensions();
    let mut out = Array3::<u32>::zeros((1, height as usize, width as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        out[[0, y as usize, x as usize]] = u32::from(pixel.0[0]);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct ForegroundOverlap {
    dice: f64,
    jaccard: f64,
}

fn foreground_overlap(
    truth: ndarray::ArrayView3<'_, u32>,
    produced: ndarray::ArrayView3<'_, u32>,
) -> Result<ForegroundOverlap> {
    if truth.shape() != produced.shape() {
        return Err(Error::ShapeMismatch {
            expected: truth.shape().to_vec(),
            got: produced.shape().to_vec(),
        });
    }
    let mut truth_count = 0u64;
    let mut produced_count = 0u64;
    let mut shared = 0u64;
    for (&truth, &produced) in truth.iter().zip(produced.iter()) {
        let truth_foreground = truth != 0;
        let produced_foreground = produced != 0;
        truth_count += u64::from(truth_foreground);
        produced_count += u64::from(produced_foreground);
        shared += u64::from(truth_foreground && produced_foreground);
    }
    let union = truth_count + produced_count - shared;
    Ok(ForegroundOverlap {
        dice: if truth_count + produced_count == 0 {
            1.0
        } else {
            2.0 * shared as f64 / (truth_count + produced_count) as f64
        },
        jaccard: if union == 0 {
            1.0
        } else {
            shared as f64 / union as f64
        },
    })
}

fn read_objects(path: &Path, columns: &ColumnConfig) -> Result<Vec<ObjectRow>> {
    let mut reader = csv::Reader::from_path(path).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-compare: could not open {}: {err}",
            path.display()
        ))
    })?;
    let headers = reader
        .headers()
        .map_err(|err| Error::invalid(format!("cellprofiler-compare: read headers: {err}")))?
        .clone();
    let bindings = ColumnBindings::new(&headers, columns)?;
    let mut rows = Vec::new();
    for (index, row) in reader.records().enumerate() {
        let row = row.map_err(|err| {
            Error::invalid(format!("cellprofiler-compare: read row {index}: {err}"))
        })?;
        rows.push(ObjectRow {
            index,
            area: bindings
                .area
                .and_then(|column| parse_optional_f64(&row, column)),
            centroid: [
                bindings
                    .centroid_z
                    .and_then(|column| parse_optional_f64(&row, column))
                    .unwrap_or(0.0),
                parse_required_f64(&row, bindings.centroid_y, "centroid_y", index)?,
                parse_required_f64(&row, bindings.centroid_x, "centroid_x", index)?,
            ],
            mean_intensity: bindings
                .mean_intensity
                .and_then(|column| parse_optional_f64(&row, column)),
            integrated_intensity: bindings
                .integrated_intensity
                .and_then(|column| parse_optional_f64(&row, column)),
            bbox_min: parse_bbox_min(&row, &bindings),
            bbox_max: parse_bbox_max(&row, &bindings),
        });
    }
    Ok(rows)
}

#[derive(Debug)]
struct ColumnBindings {
    area: Option<usize>,
    centroid_z: Option<usize>,
    centroid_y: usize,
    centroid_x: usize,
    mean_intensity: Option<usize>,
    integrated_intensity: Option<usize>,
    bbox_min_z: Option<usize>,
    bbox_min_y: Option<usize>,
    bbox_min_x: Option<usize>,
    bbox_max_z: Option<usize>,
    bbox_max_y: Option<usize>,
    bbox_max_x: Option<usize>,
}

impl ColumnBindings {
    fn new(headers: &csv::StringRecord, columns: &ColumnConfig) -> Result<Self> {
        let centroid_y = find_required(headers, columns.centroid_y.as_deref(), "centroid_y")?;
        let centroid_x = find_required(headers, columns.centroid_x.as_deref(), "centroid_x")?;
        Ok(Self {
            area: find_optional(headers, columns.area.as_deref()),
            centroid_z: find_optional(headers, columns.centroid_z.as_deref()),
            centroid_y,
            centroid_x,
            mean_intensity: find_optional(headers, columns.mean_intensity.as_deref()),
            integrated_intensity: find_optional(headers, columns.integrated_intensity.as_deref()),
            bbox_min_z: find_optional(headers, columns.bbox_min_z.as_deref()),
            bbox_min_y: find_optional(headers, columns.bbox_min_y.as_deref()),
            bbox_min_x: find_optional(headers, columns.bbox_min_x.as_deref()),
            bbox_max_z: find_optional(headers, columns.bbox_max_z.as_deref()),
            bbox_max_y: find_optional(headers, columns.bbox_max_y.as_deref()),
            bbox_max_x: find_optional(headers, columns.bbox_max_x.as_deref()),
        })
    }
}

fn parse_bbox_min(row: &csv::StringRecord, bindings: &ColumnBindings) -> Option<[f64; 3]> {
    Some([
        bindings
            .bbox_min_z
            .and_then(|column| parse_optional_f64(row, column))
            .unwrap_or(0.0),
        parse_optional_f64(row, bindings.bbox_min_y?)?,
        parse_optional_f64(row, bindings.bbox_min_x?)?,
    ])
}

fn parse_bbox_max(row: &csv::StringRecord, bindings: &ColumnBindings) -> Option<[f64; 3]> {
    Some([
        bindings
            .bbox_max_z
            .and_then(|column| parse_optional_f64(row, column))
            .unwrap_or(1.0),
        parse_optional_f64(row, bindings.bbox_max_y?)?,
        parse_optional_f64(row, bindings.bbox_max_x?)?,
    ])
}

fn find_required(
    headers: &csv::StringRecord,
    configured: Option<&str>,
    semantic: &str,
) -> Result<usize> {
    find_optional(headers, configured).ok_or_else(|| {
        Error::invalid(format!(
            "cellprofiler-compare: missing required {semantic} column {:?}; available columns: {}",
            configured,
            headers.iter().collect::<Vec<_>>().join(", ")
        ))
    })
}

fn find_optional(headers: &csv::StringRecord, configured: Option<&str>) -> Option<usize> {
    let configured = configured?;
    headers.iter().position(|header| header == configured)
}

fn parse_required_f64(
    row: &csv::StringRecord,
    column: usize,
    name: &str,
    row_index: usize,
) -> Result<f64> {
    parse_optional_f64(row, column).ok_or_else(|| {
        Error::invalid(format!(
            "cellprofiler-compare: row {row_index} has missing or invalid {name}"
        ))
    })
}

fn parse_optional_f64(row: &csv::StringRecord, column: usize) -> Option<f64> {
    let value = row.get(column)?.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("nan") {
        return None;
    }
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn match_by_centroid(
    blockflow: &[ObjectRow],
    reference: &[ObjectRow],
    max_distance: Option<f64>,
) -> Vec<MatchedPair> {
    let mut candidates = Vec::new();
    for (blockflow_index, left) in blockflow.iter().enumerate() {
        for (reference_index, right) in reference.iter().enumerate() {
            candidates.push((
                centroid_distance(left.centroid, right.centroid),
                blockflow_index,
                reference_index,
            ));
        }
    }
    candidates.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let mut used_blockflow = vec![false; blockflow.len()];
    let mut used_reference = vec![false; reference.len()];
    let mut pairs = Vec::new();
    for (distance, blockflow_index, reference_index) in candidates {
        if max_distance.is_some_and(|limit| distance > limit) {
            break;
        }
        if used_blockflow[blockflow_index] || used_reference[reference_index] {
            continue;
        }
        used_blockflow[blockflow_index] = true;
        used_reference[reference_index] = true;
        let left = &blockflow[blockflow_index];
        let right = &reference[reference_index];
        pairs.push(MatchedPair {
            blockflow: left.index,
            reference: right.index,
            centroid_distance: distance,
            area_relative_error: relative_error(left.area, right.area),
            mean_intensity_relative_error: relative_error(
                left.mean_intensity,
                right.mean_intensity,
            ),
            integrated_intensity_relative_error: relative_error(
                left.integrated_intensity,
                right.integrated_intensity,
            ),
        });
    }
    pairs.sort_by_key(|pair| pair.blockflow);
    pairs
}

fn centroid_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dz = left[0] - right[0];
    let dy = left[1] - right[1];
    let dx = left[2] - right[2];
    (dz * dz + dy * dy + dx * dx).sqrt()
}

fn relative_error(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    let left = left?;
    let right = right?;
    if right == 0.0 {
        return Some((left - right).abs());
    }
    Some((left - right).abs() / right.abs())
}

fn object_summary(row: &ObjectRow) -> serde_json::Value {
    json!({
        "row": row.index,
        "centroid": row.centroid,
        "area": row.area,
        "bbox_min": row.bbox_min,
        "bbox_max": row.bbox_max,
        "bbox_area_xy": bbox_area_xy(row),
        "bbox_diagonal_xy": bbox_diagonal_xy(row),
        "mean_intensity": row.mean_intensity,
        "integrated_intensity": row.integrated_intensity,
    })
}

fn object_with_nearest_summary(
    row: &ObjectRow,
    candidates: &[ObjectRow],
    nearest_key: &str,
    limit: usize,
) -> serde_json::Value {
    let mut summary = object_summary(row);
    if let Some(map) = summary.as_object_mut() {
        map.insert(
            nearest_key.to_string(),
            serde_json::Value::Array(nearest_objects(row, candidates, limit)),
        );
    }
    summary
}

fn nearest_objects(
    row: &ObjectRow,
    candidates: &[ObjectRow],
    limit: usize,
) -> Vec<serde_json::Value> {
    let mut ranked = candidates
        .iter()
        .map(|candidate| {
            (
                centroid_distance(row.centroid, candidate.centroid),
                candidate.index,
                candidate,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(distance, _, candidate)| {
            json!({
                "row": candidate.index,
                "centroid_distance": distance,
                "centroid": candidate.centroid,
                "area": candidate.area,
                "bbox_min": candidate.bbox_min,
                "bbox_max": candidate.bbox_max,
                "bbox_area_xy": bbox_area_xy(candidate),
                "bbox_diagonal_xy": bbox_diagonal_xy(candidate),
                "bbox_iou_xy": bbox_iou_xy(row, candidate),
                "mean_intensity": candidate.mean_intensity,
                "integrated_intensity": candidate.integrated_intensity,
            })
        })
        .collect()
}

fn split_merge_candidates(
    targets: &[ObjectRow],
    unmatched_targets: &[usize],
    parts: &[ObjectRow],
    target_key: &str,
    parts_key: &str,
) -> Vec<serde_json::Value> {
    unmatched_targets
        .iter()
        .filter_map(|&target_index| {
            let target = &targets[target_index];
            let mut overlapping = parts
                .iter()
                .filter_map(|part| {
                    let iou = bbox_iou_xy(target, part)?;
                    (iou > 0.0).then_some((
                        iou,
                        centroid_distance(target.centroid, part.centroid),
                        part,
                    ))
                })
                .collect::<Vec<_>>();
            overlapping.sort_by(|left, right| {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| left.1.total_cmp(&right.1))
                    .then_with(|| left.2.index.cmp(&right.2.index))
            });
            let selected = overlapping.into_iter().take(4).collect::<Vec<_>>();
            if selected.len() < 2 {
                return None;
            }
            let combined_area = selected
                .iter()
                .filter_map(|(_, _, part)| part.area)
                .sum::<f64>();
            let area_relative_error = relative_error(Some(combined_area), target.area);
            Some(json!({
                target_key: target.index,
                "target": object_summary(target),
                "combined_part_area": combined_area,
                "combined_area_relative_error": area_relative_error,
                parts_key: selected.into_iter().map(|(bbox_iou_xy, centroid_distance, part)| {
                    json!({
                        "row": part.index,
                        "centroid_distance": centroid_distance,
                        "bbox_iou_xy": bbox_iou_xy,
                        "area": part.area,
                        "centroid": part.centroid,
                        "bbox_min": part.bbox_min,
                        "bbox_max": part.bbox_max,
                    })
                }).collect::<Vec<_>>(),
            }))
        })
        .collect()
}

fn bbox_area_xy(row: &ObjectRow) -> Option<f64> {
    let min = row.bbox_min?;
    let max = row.bbox_max?;
    let height = (max[1] - min[1]).max(0.0);
    let width = (max[2] - min[2]).max(0.0);
    Some(height * width)
}

fn bbox_diagonal_xy(row: &ObjectRow) -> Option<f64> {
    let min = row.bbox_min?;
    let max = row.bbox_max?;
    let height = (max[1] - min[1]).max(0.0);
    let width = (max[2] - min[2]).max(0.0);
    Some((height * height + width * width).sqrt())
}

fn bbox_iou_xy(left: &ObjectRow, right: &ObjectRow) -> Option<f64> {
    let left_min = left.bbox_min?;
    let left_max = left.bbox_max?;
    let right_min = right.bbox_min?;
    let right_max = right.bbox_max?;
    let left_area = bbox_area_xy(left)?;
    let right_area = bbox_area_xy(right)?;
    let y0 = left_min[1].max(right_min[1]);
    let x0 = left_min[2].max(right_min[2]);
    let y1 = left_max[1].min(right_max[1]);
    let x1 = left_max[2].min(right_max[2]);
    let intersection = (y1 - y0).max(0.0) * (x1 - x0).max(0.0);
    let union = left_area + right_area - intersection;
    (union > 0.0).then_some(intersection / union)
}

fn match_summary(pair: &MatchedPair, value: f64) -> serde_json::Value {
    json!({
        "blockflow_row": pair.blockflow,
        "reference_row": pair.reference,
        "value": value,
        "centroid_distance": pair.centroid_distance,
        "area_relative_error": pair.area_relative_error,
        "mean_intensity_relative_error": pair.mean_intensity_relative_error,
        "integrated_intensity_relative_error": pair.integrated_intensity_relative_error,
    })
}

fn worst_matches_by(
    pairs: &[MatchedPair],
    value: impl Fn(&MatchedPair) -> Option<f64>,
    limit: usize,
) -> Vec<serde_json::Value> {
    let mut ranked = pairs
        .iter()
        .filter_map(|pair| value(pair).map(|score| (score, pair)))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.blockflow.cmp(&right.1.blockflow))
            .then_with(|| left.1.reference.cmp(&right.1.reference))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(score, pair)| match_summary(pair, score))
        .collect()
}

#[derive(Debug)]
struct ThresholdFailures {
    count: usize,
    examples: Vec<serde_json::Value>,
}

fn matches_over_threshold(
    pairs: &[MatchedPair],
    value: impl Fn(&MatchedPair) -> Option<f64>,
    threshold: f64,
    limit: usize,
) -> ThresholdFailures {
    let mut failures = pairs
        .iter()
        .filter_map(|pair| {
            let score = value(pair)?;
            (score > threshold).then_some((score, pair))
        })
        .collect::<Vec<_>>();
    failures.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.blockflow.cmp(&right.1.blockflow))
            .then_with(|| left.1.reference.cmp(&right.1.reference))
    });
    ThresholdFailures {
        count: failures.len(),
        examples: failures
            .into_iter()
            .take(limit)
            .map(|(score, pair)| match_summary(pair, score))
            .collect(),
    }
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count != 0).then_some(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid_matching_is_greedy_and_one_to_one() {
        let blockflow = vec![
            row(0, [0.0, 10.0, 10.0], Some(12.0), Some(5.0)),
            row(1, [0.0, 40.0, 40.0], Some(20.0), Some(7.0)),
        ];
        let reference = vec![
            row(0, [0.0, 41.0, 40.0], Some(21.0), Some(7.0)),
            row(1, [0.0, 9.0, 10.0], Some(12.0), Some(5.0)),
        ];
        let pairs = match_by_centroid(&blockflow, &reference, None);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].blockflow, 0);
        assert_eq!(pairs[0].reference, 1);
        assert_eq!(pairs[1].blockflow, 1);
        assert_eq!(pairs[1].reference, 0);
    }

    #[test]
    fn relative_error_is_reported_when_both_columns_exist() {
        let blockflow = vec![row(0, [0.0, 0.0, 0.0], Some(11.0), Some(8.0))];
        let reference = vec![row(0, [0.0, 0.0, 0.0], Some(10.0), Some(10.0))];
        let pairs = match_by_centroid(&blockflow, &reference, None);
        assert_eq!(pairs[0].area_relative_error, Some(0.1));
        assert_eq!(pairs[0].mean_intensity_relative_error, Some(0.2));
    }

    #[test]
    fn centroid_matching_can_gate_distant_objects() {
        let blockflow = vec![
            row(0, [0.0, 10.0, 10.0], Some(12.0), Some(5.0)),
            row(1, [0.0, 100.0, 100.0], Some(20.0), Some(7.0)),
        ];
        let reference = vec![
            row(0, [0.0, 11.0, 10.0], Some(12.0), Some(5.0)),
            row(1, [0.0, 200.0, 200.0], Some(20.0), Some(7.0)),
        ];
        let pairs = match_by_centroid(&blockflow, &reference, Some(5.0));
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].blockflow, 0);
        assert_eq!(pairs[0].reference, 0);
    }

    #[test]
    fn cellprofiler_defaults_include_dna_intensity_columns() {
        let defaults = ColumnConfig::cellprofiler_defaults();
        assert_eq!(
            defaults.mean_intensity.as_deref(),
            Some("Intensity_MeanIntensity_DNA")
        );
        assert_eq!(
            defaults.integrated_intensity.as_deref(),
            Some("Intensity_IntegratedIntensity_DNA")
        );
    }

    #[test]
    fn worst_match_diagnostics_are_sorted_descending() {
        let pairs = vec![
            MatchedPair {
                blockflow: 0,
                reference: 0,
                centroid_distance: 1.0,
                area_relative_error: Some(0.1),
                mean_intensity_relative_error: None,
                integrated_intensity_relative_error: None,
            },
            MatchedPair {
                blockflow: 1,
                reference: 1,
                centroid_distance: 4.0,
                area_relative_error: Some(0.5),
                mean_intensity_relative_error: None,
                integrated_intensity_relative_error: None,
            },
            MatchedPair {
                blockflow: 2,
                reference: 2,
                centroid_distance: 3.0,
                area_relative_error: Some(0.2),
                mean_intensity_relative_error: None,
                integrated_intensity_relative_error: None,
            },
        ];
        let worst = worst_matches_by(&pairs, |pair| pair.area_relative_error, 2);
        assert_eq!(worst[0]["blockflow_row"], 1);
        assert_eq!(worst[0]["value"], 0.5);
        assert_eq!(worst[1]["blockflow_row"], 2);
    }

    #[test]
    fn nearest_object_diagnostics_are_sorted_by_distance() {
        let target = row(9, [0.0, 10.0, 10.0], Some(30.0), Some(0.4));
        let candidates = vec![
            row(0, [0.0, 20.0, 10.0], Some(100.0), Some(0.1)),
            row(1, [0.0, 11.0, 10.0], Some(50.0), Some(0.2)),
            row(2, [0.0, 10.0, 13.0], Some(60.0), Some(0.3)),
        ];
        let nearest = nearest_objects(&target, &candidates, 2);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0]["row"], 1);
        assert_eq!(nearest[0]["centroid_distance"], 1.0);
        assert_eq!(nearest[1]["row"], 2);
        assert_eq!(nearest[1]["centroid_distance"], 3.0);
    }

    #[test]
    fn split_merge_candidates_group_overlapping_parts() {
        let targets = vec![row_with_bbox(
            9,
            [0.0, 10.0, 10.0],
            Some(100.0),
            Some(0.4),
            [0.0, 0.0, 0.0],
            [1.0, 10.0, 10.0],
        )];
        let parts = vec![
            row_with_bbox(
                1,
                [0.0, 4.0, 4.0],
                Some(40.0),
                Some(0.2),
                [0.0, 0.0, 0.0],
                [1.0, 6.0, 6.0],
            ),
            row_with_bbox(
                2,
                [0.0, 7.0, 7.0],
                Some(60.0),
                Some(0.3),
                [0.0, 4.0, 4.0],
                [1.0, 10.0, 10.0],
            ),
            row_with_bbox(
                3,
                [0.0, 30.0, 30.0],
                Some(10.0),
                Some(0.3),
                [0.0, 25.0, 25.0],
                [1.0, 30.0, 30.0],
            ),
        ];
        let candidates =
            split_merge_candidates(&targets, &[0], &parts, "reference_row", "blockflow_parts");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0]["reference_row"], 9);
        assert_eq!(candidates[0]["combined_part_area"], 100.0);
        assert_eq!(candidates[0]["combined_area_relative_error"], 0.0);
        assert_eq!(
            candidates[0]["blockflow_parts"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn threshold_failures_count_all_and_limit_examples() {
        let pairs = vec![
            MatchedPair {
                blockflow: 0,
                reference: 0,
                centroid_distance: 2.0,
                area_relative_error: None,
                mean_intensity_relative_error: None,
                integrated_intensity_relative_error: None,
            },
            MatchedPair {
                blockflow: 1,
                reference: 1,
                centroid_distance: 8.0,
                area_relative_error: None,
                mean_intensity_relative_error: None,
                integrated_intensity_relative_error: None,
            },
            MatchedPair {
                blockflow: 2,
                reference: 2,
                centroid_distance: 6.0,
                area_relative_error: None,
                mean_intensity_relative_error: None,
                integrated_intensity_relative_error: None,
            },
        ];
        let failures = matches_over_threshold(&pairs, |pair| Some(pair.centroid_distance), 5.0, 1);
        assert_eq!(failures.count, 2);
        assert_eq!(failures.examples.len(), 1);
        assert_eq!(failures.examples[0]["blockflow_row"], 1);
        assert_eq!(failures.examples[0]["value"], 8.0);
    }

    #[test]
    fn foreground_overlap_reports_dice_and_jaccard() {
        let truth = Array3::from_shape_vec((1, 2, 3), vec![0, 1, 1, 0, 2, 0]).unwrap();
        let produced = Array3::from_shape_vec((1, 2, 3), vec![0, 3, 0, 0, 4, 4]).unwrap();
        let overlap = foreground_overlap(truth.view(), produced.view()).unwrap();
        assert_eq!(overlap.dice, 4.0 / 6.0);
        assert_eq!(overlap.jaccard, 2.0 / 4.0);
    }

    #[test]
    fn bbox_diagnostics_report_xy_area_and_iou() {
        let left = row_with_bbox(
            0,
            [0.0, 5.0, 5.0],
            Some(100.0),
            Some(0.1),
            [0.0, 0.0, 0.0],
            [1.0, 10.0, 10.0],
        );
        let right = row_with_bbox(
            1,
            [0.0, 10.0, 10.0],
            Some(100.0),
            Some(0.1),
            [0.0, 5.0, 5.0],
            [1.0, 15.0, 15.0],
        );
        assert_eq!(bbox_area_xy(&left), Some(100.0));
        assert_eq!(bbox_iou_xy(&left, &right), Some(25.0 / 175.0));
    }

    fn row(
        index: usize,
        centroid: [f64; 3],
        area: Option<f64>,
        mean_intensity: Option<f64>,
    ) -> ObjectRow {
        ObjectRow {
            index,
            area,
            centroid,
            mean_intensity,
            integrated_intensity: None,
            bbox_min: None,
            bbox_max: None,
        }
    }

    fn row_with_bbox(
        index: usize,
        centroid: [f64; 3],
        area: Option<f64>,
        mean_intensity: Option<f64>,
        bbox_min: [f64; 3],
        bbox_max: [f64; 3],
    ) -> ObjectRow {
        ObjectRow {
            index,
            area,
            centroid,
            mean_intensity,
            integrated_intensity: None,
            bbox_min: Some(bbox_min),
            bbox_max: Some(bbox_max),
        }
    }
}
