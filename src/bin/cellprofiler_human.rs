// SPDX-License-Identifier: MIT

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::ops::{
    distance_transform, fill_label_holes_2d_by_label_into,
    filter_labels_touching_border_on_axes_into, gaussian_smooth_into_with, li_threshold,
    otsu_threshold, regional_maxima, remove_small_objects_into, seeded_watershed, Boundary,
    Connectivity, DistanceParams, Gaussian, IntensityMeasurements, RegionShape, Separation,
    ShapeMeasurements, PAIRS,
};
use blockflow::{Error, Result};
use image::{ImageBuffer, Luma};
use ndarray::Array3;

#[derive(Debug)]
struct Config {
    input: PathBuf,
    out_dir: PathBuf,
    sigma: f64,
    declump_sigma: f64,
    threshold_method: ThresholdMethod,
    threshold_bins: usize,
    min_size: u64,
    max_size: Option<u64>,
    seed_min_distance: f64,
    maxima_downsample: usize,
    declump_method: DeclumpMethod,
    fill_holes_after_declumping: bool,
    merge_line_basin_pixels: usize,
    merge_line_max_saddle_drop: Option<f64>,
    background_percentile: Option<f64>,
    keep_border: bool,
    watershed_line: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ThresholdMethod {
    Li,
    Otsu,
}

impl ThresholdMethod {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "li" | "minimum-cross-entropy" | "minimum_cross_entropy" => Ok(Self::Li),
            "otsu" => Ok(Self::Otsu),
            other => Err(Error::invalid(format!(
                "cellprofiler-human: unknown --threshold-method {other:?}; expected li or otsu"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Li => "li",
            Self::Otsu => "otsu",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DeclumpMethod {
    Intensity,
    Distance,
}

impl DeclumpMethod {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "intensity" => Ok(Self::Intensity),
            "distance" | "shape" => Ok(Self::Distance),
            other => Err(Error::invalid(format!(
                "cellprofiler-human: unknown --declump-method {other:?}; expected intensity or distance"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Intensity => "intensity",
            Self::Distance => "distance",
        }
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let mut input = None;
        let mut out_dir = None;
        let mut sigma = 1.0;
        let mut declump_sigma = 1.3488;
        let mut threshold_method = ThresholdMethod::Li;
        let mut threshold_bins = 256usize;
        let mut min_size = 50u64;
        let mut max_size = Some(5027u64);
        let mut seed_min_distance = 6.0;
        let mut maxima_downsample = 3usize;
        let mut declump_method = DeclumpMethod::Intensity;
        let mut fill_holes_after_declumping = true;
        let mut merge_line_basin_pixels = 0usize;
        let mut merge_line_max_saddle_drop = None;
        let mut background_percentile = Some(1.0);
        let mut keep_border = false;
        let mut watershed_line = true;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = Some(path_arg(&mut args, "--input")?),
                "--out" => out_dir = Some(path_arg(&mut args, "--out")?),
                "--sigma" => sigma = parse_arg(&mut args, "--sigma")?,
                "--declump-sigma" => declump_sigma = parse_arg(&mut args, "--declump-sigma")?,
                "--threshold-method" => {
                    threshold_method =
                        ThresholdMethod::parse(&string_arg(&mut args, "--threshold-method")?)?
                }
                "--threshold-bins" => threshold_bins = parse_arg(&mut args, "--threshold-bins")?,
                "--min-size" => min_size = parse_arg(&mut args, "--min-size")?,
                "--max-size" => max_size = Some(parse_arg(&mut args, "--max-size")?),
                "--no-max-size" => max_size = None,
                "--seed-min-distance" => {
                    seed_min_distance = parse_arg(&mut args, "--seed-min-distance")?
                }
                "--maxima-downsample" => {
                    maxima_downsample = parse_arg(&mut args, "--maxima-downsample")?
                }
                "--declump-method" => {
                    declump_method =
                        DeclumpMethod::parse(&string_arg(&mut args, "--declump-method")?)?
                }
                "--no-fill-holes-after-declumping" => fill_holes_after_declumping = false,
                "--merge-line-basin-pixels" => {
                    merge_line_basin_pixels = parse_arg(&mut args, "--merge-line-basin-pixels")?
                }
                "--merge-line-max-saddle-drop" => {
                    merge_line_max_saddle_drop =
                        Some(parse_arg(&mut args, "--merge-line-max-saddle-drop")?)
                }
                "--background-percentile" => {
                    background_percentile = Some(parse_arg(&mut args, "--background-percentile")?)
                }
                "--no-background-subtract" => background_percentile = None,
                "--keep-border" => keep_border = true,
                "--adjacent-basins" => watershed_line = false,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::invalid(format!(
                        "cellprofiler-human: unknown argument {other:?}; use --help"
                    )));
                }
            }
        }

        Self {
            input: input.ok_or_else(|| {
                Error::invalid("cellprofiler-human: missing required --input IMAGE")
            })?,
            out_dir: out_dir.unwrap_or_else(|| PathBuf::from(".tmp/cellprofiler-human/run")),
            sigma,
            declump_sigma,
            threshold_method,
            threshold_bins,
            min_size,
            max_size,
            seed_min_distance,
            maxima_downsample,
            declump_method,
            fill_holes_after_declumping,
            merge_line_basin_pixels,
            merge_line_max_saddle_drop,
            background_percentile,
            keep_border,
            watershed_line,
        }
        .validate()
    }

    fn validate(self) -> Result<Self> {
        if self.sigma < 0.0 || !self.sigma.is_finite() {
            return Err(Error::invalid(
                "cellprofiler-human: --sigma must be finite and non-negative",
            ));
        }
        if self.declump_sigma < 0.0 || !self.declump_sigma.is_finite() {
            return Err(Error::invalid(
                "cellprofiler-human: --declump-sigma must be finite and non-negative",
            ));
        }
        if self.threshold_bins < 2 {
            return Err(Error::invalid(
                "cellprofiler-human: --threshold-bins must be at least 2",
            ));
        }
        if self.seed_min_distance < 0.0 || !self.seed_min_distance.is_finite() {
            return Err(Error::invalid(
                "cellprofiler-human: --seed-min-distance must be finite and non-negative",
            ));
        }
        if self.maxima_downsample == 0 {
            return Err(Error::invalid(
                "cellprofiler-human: --maxima-downsample must be at least 1",
            ));
        }
        if self
            .merge_line_max_saddle_drop
            .is_some_and(|value| !value.is_finite())
        {
            return Err(Error::invalid(
                "cellprofiler-human: --merge-line-max-saddle-drop must be finite",
            ));
        }
        if let Some(max_size) = self.max_size {
            if max_size < self.min_size {
                return Err(Error::invalid(
                    "cellprofiler-human: --max-size must be at least --min-size",
                ));
            }
        }
        Ok(self)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    fs::create_dir_all(&config.out_dir).map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-human: could not create {}: {err}",
            config.out_dir.display()
        ))
    })?;

    let started = Instant::now();
    let input = load_luma_as_volume(&config.input)?;
    let loaded = started.elapsed();

    let pipeline_started = Instant::now();
    let measurement_values = normalized_luma_measurement_values(input.view());
    let corrected = subtract_background(input, config.background_percentile);
    let smoothed = smooth_xy(corrected.view(), config.sigma)?;
    let declump_smoothed = if config.declump_method == DeclumpMethod::Intensity
        && config.declump_sigma.to_bits() != config.sigma.to_bits()
    {
        Some(smooth_xy(corrected.view(), config.declump_sigma)?)
    } else {
        None
    };
    let threshold = threshold_from_volume(
        smoothed.view(),
        config.threshold_method,
        config.threshold_bins,
    )?;
    let raw_mask = smoothed.mapv(|value| value > threshold);

    let mut sized_mask = Array3::<bool>::from_elem(raw_mask.raw_dim(), false);
    remove_small_objects_into(
        raw_mask.view(),
        Connectivity::Faces,
        config.min_size,
        sized_mask.view_mut(),
    )?;

    let final_mask = sized_mask;

    let distances;
    let declump_source = match config.declump_method {
        DeclumpMethod::Intensity => declump_smoothed
            .as_ref()
            .map_or_else(|| smoothed.view(), |image| image.view()),
        DeclumpMethod::Distance => {
            distances = distance_transform(final_mask.view(), &DistanceParams::default())?;
            distances.view()
        }
    };
    let maxima = regional_maxima_for_seeding(declump_source, config.maxima_downsample)?;
    let (mut seeds, seed_count) = maxima_seeds(
        maxima.view(),
        declump_source,
        final_mask.view(),
        config.seed_min_distance,
    )?;
    if seed_count == 0 && final_mask.iter().any(|&inside| inside) {
        blockflow::ops::components::label_members_into_with(
            shape_of(&final_mask),
            Connectivity::Faces,
            |at| final_mask[[at[0], at[1], at[2]]],
            seeds.view_mut(),
        )?;
    }

    let cost = declump_source.mapv(|value| -value);
    let labels = seeded_watershed(
        cost.view(),
        seeds.view(),
        Some(final_mask.view()),
        if config.watershed_line {
            Separation::Line
        } else {
            Separation::Adjacent
        },
    )?;

    let labels = if config.fill_holes_after_declumping {
        fill_label_holes_after_declumping(labels.view())?
    } else {
        labels
    };
    let labels = if config.merge_line_basin_pixels == 0 {
        labels
    } else {
        merge_labels_across_watershed_lines(
            labels.view(),
            declump_source,
            config.merge_line_basin_pixels,
            config.merge_line_max_saddle_drop,
        )?
    };
    let labels = if config.keep_border {
        labels
    } else {
        filter_labels_touching_border_for_input_dimensionality(labels.view())?
    };
    let labels = filter_labels_by_size(labels.view(), config.min_size, config.max_size)?;
    let rows = measure_objects(labels.view(), measurement_values.view())?;
    let pipeline_elapsed = pipeline_started.elapsed();

    save_mask(&final_mask, &config.out_dir.join("foreground_mask.png"))?;
    save_labels(&labels, &config.out_dir.join("labels.png"))?;
    write_object_csv(&rows, &config.out_dir.join("objects.csv"))?;
    write_summary_json(
        &config,
        &rows,
        threshold,
        loaded.as_secs_f64(),
        pipeline_elapsed.as_secs_f64(),
        &config.out_dir.join("summary.json"),
    )?;

    println!(
        "objects={} threshold={threshold:.6} output={}",
        rows.len(),
        config.out_dir.display()
    );
    Ok(())
}

fn print_help() {
    println!(
        "cellprofiler-human --input IMAGE [--out DIR]\n\
         \n\
         Runs a small CellProfiler-style nuclei/cell segmentation benchmark path:\n\
          background subtraction, XY Gaussian smoothing, Li/Otsu threshold,\n\
           mask cleanup, distance-transform watershed, and object measurements.\n\
         \n\
         Options:\n\
           --sigma FLOAT                    XY Gaussian sigma, default 1.0\n\
           --declump-sigma FLOAT            intensity declumping Gaussian sigma, default 1.3488\n\
           --threshold-method li|otsu       global threshold method, default li\n\
           --threshold-bins N               Otsu histogram bins, default 256\n\
           --min-size N                     remove objects below N voxels, default 50\n\
           --max-size N                     remove objects above N voxels, default 5027\n\
           --no-max-size                    disable large-object filtering\n\
           --seed-min-distance FLOAT        suppress watershed seeds closer than this, default 6\n\
           --maxima-downsample N            find seed maxima on lower-resolution XY blocks, default 3\n\
           --declump-method intensity|distance watershed source, default intensity\n\
           --no-fill-holes-after-declumping skip CellProfiler-style post-declump hole filling\n\
           --merge-line-basin-pixels N      merge labels separated by at least N watershed-line pixels\n\
           --merge-line-max-saddle-drop N   require weak-boundary mean minus line mean to be at most N\n\
           --background-percentile FLOAT    subtract this percentile, default 1.0\n\
           --no-background-subtract         leave intensities unchanged before smoothing\n\
           --keep-border                    keep objects touching the image border\n\
           --adjacent-basins                watershed basins touch instead of carving lines"
    );
}

fn path_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(args.next().ok_or_else(|| {
        Error::invalid(format!("cellprofiler-human: {name} needs a path"))
    })?))
}

fn string_arg(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| Error::invalid(format!("cellprofiler-human: {name} needs a value")))
}

fn parse_arg<T>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| Error::invalid(format!("cellprofiler-human: {name} needs a value")))?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "cellprofiler-human: could not parse {name} value {raw:?}: {err}"
        ))
    })
}

fn load_luma_as_volume(path: &Path) -> Result<Array3<f64>> {
    let image = image::ImageReader::open(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: open image: {err}")))?
        .decode()
        .map_err(|err| Error::invalid(format!("cellprofiler-human: decode image: {err}")))?
        .to_luma16();
    let (width, height) = image.dimensions();
    let width = usize::try_from(width)
        .map_err(|_| Error::invalid("cellprofiler-human: image width does not fit usize"))?;
    let height = usize::try_from(height)
        .map_err(|_| Error::invalid("cellprofiler-human: image height does not fit usize"))?;
    let mut out = Array3::<f64>::zeros((1, height, width));
    for (x, y, pixel) in image.enumerate_pixels() {
        out[[0, y as usize, x as usize]] = f64::from(pixel.0[0]);
    }
    Ok(out)
}

fn subtract_background(mut image: Array3<f64>, percentile: Option<f64>) -> Array3<f64> {
    let Some(percentile) = percentile else {
        return image;
    };
    let level = percentile_value(
        image.iter().copied().filter(|value| value.is_finite()),
        percentile,
    );
    for value in &mut image {
        *value = (*value - level).max(0.0);
    }
    image
}

fn normalized_luma_measurement_values(input: ndarray::ArrayView3<'_, f64>) -> Array3<f64> {
    input.mapv(|value| value / 65535.0)
}

fn percentile_value(values: impl Iterator<Item = f64>, percentile: f64) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let percentile = percentile.clamp(0.0, 100.0);
    let rank = ((percentile / 100.0) * (values.len().saturating_sub(1)) as f64).round() as usize;
    values[rank]
}

fn smooth_xy(input: ndarray::ArrayView3<'_, f64>, sigma: f64) -> Result<Array3<f64>> {
    let gaussian = Gaussian::new([0.0, sigma, sigma], 3.0)?;
    let mut out = Array3::<f64>::zeros(input.raw_dim());
    gaussian_smooth_into_with(input, gaussian.kernels(), Boundary::Reflect, out.view_mut())?;
    Ok(out)
}

fn threshold_from_volume(
    values: ndarray::ArrayView3<'_, f64>,
    method: ThresholdMethod,
    bins: usize,
) -> Result<f64> {
    let flat = values.iter().copied().collect::<Vec<_>>();
    match method {
        ThresholdMethod::Li => li_threshold(&flat),
        ThresholdMethod::Otsu => otsu_threshold(&flat, bins),
    }
}

fn maxima_seeds(
    maxima: ndarray::ArrayView3<'_, bool>,
    distances: ndarray::ArrayView3<'_, f64>,
    mask: ndarray::ArrayView3<'_, bool>,
    min_distance: f64,
) -> Result<(Array3<u32>, u32)> {
    if maxima.shape() != distances.shape() || maxima.shape() != mask.shape() {
        return Err(Error::invalid(
            "cellprofiler-human: maxima, distance and mask shape mismatch",
        ));
    }
    if min_distance == 0.0 {
        let mut seeds = Array3::<u32>::zeros(maxima.raw_dim());
        let count = blockflow::ops::components::label_members_into_with(
            view_shape(&maxima),
            Connectivity::Faces,
            |at| maxima[[at[0], at[1], at[2]]] && mask[[at[0], at[1], at[2]]],
            seeds.view_mut(),
        )?;
        return Ok((seeds, count));
    }

    let mut candidates = Vec::<([usize; 3], f64)>::new();
    for ((z, y, x), &is_maximum) in maxima.indexed_iter() {
        if is_maximum && mask[[z, y, x]] {
            candidates.push(([z, y, x], distances[[z, y, x]]));
        }
    }
    candidates.sort_by(|(a_at, a_score), (b_at, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a_at.cmp(b_at))
    });

    let min_distance2 = min_distance * min_distance;
    let mut accepted = Vec::<[usize; 3]>::new();
    for (at, _) in candidates {
        if accepted
            .iter()
            .all(|&seed| squared_distance(at, seed) >= min_distance2)
        {
            accepted.push(at);
        }
    }

    let seed_count = u32::try_from(accepted.len())
        .map_err(|_| Error::invalid("cellprofiler-human: watershed seed count does not fit u32"))?;
    let mut seeds = Array3::<u32>::zeros(maxima.raw_dim());
    for (index, at) in accepted.into_iter().enumerate() {
        seeds[[at[0], at[1], at[2]]] = u32::try_from(index + 1)
            .map_err(|_| Error::invalid("cellprofiler-human: seed index overflow"))?;
    }
    Ok((seeds, seed_count))
}

fn regional_maxima_for_seeding(
    values: ndarray::ArrayView3<'_, f64>,
    downsample: usize,
) -> Result<Array3<bool>> {
    if downsample <= 1 {
        return regional_maxima(values);
    }
    let shape = view_shape(&values);
    let low_y = shape[1].div_ceil(downsample);
    let low_x = shape[2].div_ceil(downsample);
    let mut low = Array3::<f64>::from_elem((shape[0], low_y, low_x), f64::NEG_INFINITY);
    for ((z, y, x), &value) in values.indexed_iter() {
        if value.is_finite() {
            let slot = &mut low[[z, y / downsample, x / downsample]];
            *slot = slot.max(value);
        }
    }
    let low_maxima = regional_maxima(low.view())?;
    let mut high = Array3::<bool>::from_elem(values.raw_dim(), false);
    for ((z, low_y, low_x), &is_maximum) in low_maxima.indexed_iter() {
        if !is_maximum {
            continue;
        }
        let y0 = low_y * downsample;
        let x0 = low_x * downsample;
        let y1 = (y0 + downsample).min(shape[1]);
        let x1 = (x0 + downsample).min(shape[2]);
        let mut best = None::<([usize; 3], f64)>;
        for y in y0..y1 {
            for x in x0..x1 {
                let value = values[[z, y, x]];
                if !value.is_finite() {
                    continue;
                }
                match best {
                    Some((best_at, best_value))
                        if value < best_value || (value == best_value && [z, y, x] >= best_at) => {}
                    _ => best = Some(([z, y, x], value)),
                }
            }
        }
        if let Some((at, _)) = best {
            high[[at[0], at[1], at[2]]] = true;
        }
    }
    Ok(high)
}

fn squared_distance(a: [usize; 3], b: [usize; 3]) -> f64 {
    let dz = a[0] as f64 - b[0] as f64;
    let dy = a[1] as f64 - b[1] as f64;
    let dx = a[2] as f64 - b[2] as f64;
    dz * dz + dy * dy + dx * dx
}

fn filter_labels_by_size(
    labels: ndarray::ArrayView3<'_, u32>,
    min_size: u64,
    max_size: Option<u64>,
) -> Result<Array3<u32>> {
    let mut counts = BTreeMap::<u32, u64>::new();
    for &label in labels.iter() {
        if label != 0 {
            let count = counts.entry(label).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                Error::invalid("cellprofiler-human: label voxel count overflowed")
            })?;
        }
    }
    let mut out = Array3::<u32>::zeros(labels.raw_dim());
    for ((z, y, x), slot) in out.indexed_iter_mut() {
        let label = labels[[z, y, x]];
        let Some(&count) = counts.get(&label) else {
            continue;
        };
        if count >= min_size && max_size.is_none_or(|limit| count <= limit) {
            *slot = label;
        }
    }
    Ok(out)
}

fn fill_label_holes_after_declumping(labels: ndarray::ArrayView3<'_, u32>) -> Result<Array3<u32>> {
    if labels.shape()[0] != 1 {
        return Err(Error::invalid(
            "cellprofiler-human: post-declump hole filling is currently implemented for 2-D images",
        ));
    }
    let mut out = Array3::<u32>::zeros(labels.raw_dim());
    fill_label_holes_2d_by_label_into(labels, out.view_mut())?;
    Ok(out)
}

fn merge_labels_across_watershed_lines(
    labels: ndarray::ArrayView3<'_, u32>,
    intensity: ndarray::ArrayView3<'_, f64>,
    min_line_pixels: usize,
    max_saddle_drop: Option<f64>,
) -> Result<Array3<u32>> {
    if labels.shape() != intensity.shape() {
        return Err(Error::invalid(
            "cellprofiler-human: line-merge labels and intensity shape mismatch",
        ));
    }
    let mut pair_evidence = BTreeMap::<(u32, u32), LineMergeEvidence>::new();
    for ((z, y, x), &label) in labels.indexed_iter() {
        if label != 0 {
            continue;
        }
        let mut touching = BTreeSet::<u32>::new();
        let mut neighbours = Vec::<(u32, f64)>::new();
        for [dz, dy, dx] in [
            [-1isize, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ] {
            let Some(nz) = z.checked_add_signed(dz) else {
                continue;
            };
            let Some(ny) = y.checked_add_signed(dy) else {
                continue;
            };
            let Some(nx) = x.checked_add_signed(dx) else {
                continue;
            };
            if nz >= labels.shape()[0] || ny >= labels.shape()[1] || nx >= labels.shape()[2] {
                continue;
            }
            let neighbour = labels[[nz, ny, nx]];
            if neighbour != 0 {
                touching.insert(neighbour);
                neighbours.push((neighbour, intensity[[nz, ny, nx]]));
            }
        }
        if touching.len() == 2 {
            let mut touching_labels = touching.into_iter();
            let a = touching_labels.next().expect("two touching labels");
            let b = touching_labels.next().expect("two touching labels");
            let pair = if a < b { (a, b) } else { (b, a) };
            pair_evidence
                .entry(pair)
                .or_default()
                .add(intensity[[z, y, x]], neighbours);
        }
    }
    let mut parent = BTreeMap::<u32, u32>::new();
    for ((a, b), evidence) in pair_evidence {
        if evidence.line_pixels >= min_line_pixels
            && evidence
                .saddle_drop()
                .is_none_or(|drop| max_saddle_drop.is_none_or(|limit| drop <= limit))
        {
            union_label(&mut parent, a, b);
        }
    }
    let mut out = Array3::<u32>::zeros(labels.raw_dim());
    for ((z, y, x), slot) in out.indexed_iter_mut() {
        let label = labels[[z, y, x]];
        if label != 0 {
            *slot = find_label(&mut parent, label);
        }
    }
    Ok(out)
}

#[derive(Default)]
struct LineMergeEvidence {
    line_pixels: usize,
    line_sum: f64,
    boundary: BTreeMap<u32, (f64, usize)>,
}

impl LineMergeEvidence {
    fn add(&mut self, line_value: f64, neighbours: Vec<(u32, f64)>) {
        self.line_pixels += 1;
        self.line_sum += line_value;
        for (label, value) in neighbours {
            let entry = self.boundary.entry(label).or_default();
            entry.0 += value;
            entry.1 += 1;
        }
    }

    fn saddle_drop(&self) -> Option<f64> {
        if self.line_pixels == 0 || self.boundary.len() != 2 {
            return None;
        }
        let line_mean = self.line_sum / self.line_pixels as f64;
        let weakest_boundary_mean = self
            .boundary
            .values()
            .filter_map(|(sum, count)| (*count > 0).then_some(*sum / *count as f64))
            .min_by(f64::total_cmp)?;
        Some(weakest_boundary_mean - line_mean)
    }
}

fn find_label(parent: &mut BTreeMap<u32, u32>, label: u32) -> u32 {
    let current = parent.get(&label).copied().unwrap_or(label);
    if current == label {
        parent.entry(label).or_insert(label);
        return label;
    }
    let root = find_label(parent, current);
    parent.insert(label, root);
    root
}

fn union_label(parent: &mut BTreeMap<u32, u32>, a: u32, b: u32) {
    let root_a = find_label(parent, a);
    let root_b = find_label(parent, b);
    if root_a == root_b {
        return;
    }
    let root = root_a.min(root_b);
    let other = root_a.max(root_b);
    parent.insert(other, root);
}

fn filter_labels_touching_border_for_input_dimensionality(
    labels: ndarray::ArrayView3<'_, u32>,
) -> Result<Array3<u32>> {
    let axes = if labels.shape()[0] == 1 {
        [false, true, true]
    } else {
        [true, true, true]
    };
    let mut out = Array3::<u32>::zeros(labels.raw_dim());
    filter_labels_touching_border_on_axes_into(labels, axes, out.view_mut())?;
    Ok(out)
}

#[derive(Debug)]
struct ObjectRow {
    label: u64,
    shape: ShapeMeasurements,
    intensity: IntensityMeasurements,
}

#[derive(Debug, Clone)]
struct ObjectTally {
    count: u64,
    position: [u64; 3],
    second: [i128; 6],
    bbox_min: [u64; 3],
    bbox_max: [u64; 3],
    nonfinite: u64,
    sum: f64,
    min: f64,
    max: f64,
    weighted_position: [f64; 3],
}

impl ObjectTally {
    fn new(at: [usize; 3], value: f64) -> Self {
        let mut tally = Self {
            count: 0,
            position: [0; 3],
            second: [0; 6],
            bbox_min: [u64::MAX; 3],
            bbox_max: [0; 3],
            nonfinite: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            weighted_position: [0.0; 3],
        };
        tally.add(at, value);
        tally
    }

    fn add(&mut self, at: [usize; 3], value: f64) {
        self.count += 1;
        let coords = [at[0] as u64, at[1] as u64, at[2] as u64];
        for axis in 0..3 {
            self.position[axis] += coords[axis];
            self.bbox_min[axis] = self.bbox_min[axis].min(coords[axis]);
            self.bbox_max[axis] = self.bbox_max[axis].max(coords[axis] + 1);
        }
        for (slot, [a, b]) in self.second.iter_mut().zip(PAIRS) {
            *slot += coords[a] as i128 * coords[b] as i128;
        }
        if value.is_finite() {
            self.sum += value;
            self.min = self.min.min(value);
            self.max = self.max.max(value);
            for axis in 0..3 {
                self.weighted_position[axis] += value * coords[axis] as f64;
            }
        } else {
            self.nonfinite += 1;
        }
    }

    fn shape(&self, label: u64) -> Result<RegionShape> {
        let at = rounded_centroid(self.position, self.count)
            .ok_or_else(|| Error::invalid("cellprofiler-human: empty object tally"))?;
        let mut central = [0i64; 6];
        for (index, [a, b]) in PAIRS.into_iter().enumerate() {
            let centre_a = at[a] as i128;
            let centre_b = at[b] as i128;
            let value = self.second[index]
                - centre_a * self.position[b] as i128
                - centre_b * self.position[a] as i128
                + self.count as i128 * centre_a * centre_b;
            central[index] = i64::try_from(value).map_err(|_| {
                Error::invalid("cellprofiler-human: central moment does not fit i64")
            })?;
        }
        Ok(RegionShape {
            label,
            at,
            count: self.count,
            position: self.position,
            central,
            bbox_min: self.bbox_min,
            bbox_max: self.bbox_max,
        })
    }

    fn intensity(&self, label: u64, at: [usize; 3]) -> IntensityMeasurements {
        let finite_count = self.count - self.nonfinite;
        IntensityMeasurements {
            label,
            count: self.count,
            finite_count,
            nonfinite: self.nonfinite,
            sum: self.sum,
            mean: (finite_count != 0).then_some(self.sum / finite_count as f64),
            min: (finite_count != 0).then_some(self.min),
            max: (finite_count != 0).then_some(self.max),
            weighted_centroid: (self.sum != 0.0).then_some([
                self.weighted_position[0] / self.sum,
                self.weighted_position[1] / self.sum,
                self.weighted_position[2] / self.sum,
            ]),
        }
        .with_fallback_centroid(at)
    }
}

trait IntensityFallback {
    fn with_fallback_centroid(self, _at: [usize; 3]) -> Self;
}

impl IntensityFallback for IntensityMeasurements {
    fn with_fallback_centroid(self, _at: [usize; 3]) -> Self {
        self
    }
}

fn measure_objects(
    labels: ndarray::ArrayView3<'_, u32>,
    values: ndarray::ArrayView3<'_, f64>,
) -> Result<Vec<ObjectRow>> {
    if labels.shape() != values.shape() {
        return Err(Error::invalid(
            "cellprofiler-human: label/value shape mismatch",
        ));
    }
    let mut tallies = BTreeMap::<u64, ObjectTally>::new();
    for ((z, y, x), &raw_label) in labels.indexed_iter() {
        if raw_label == 0 {
            continue;
        }
        let label = u64::from(raw_label);
        let at = [z, y, x];
        let value = values[[z, y, x]];
        tallies
            .entry(label)
            .and_modify(|tally| tally.add(at, value))
            .or_insert_with(|| ObjectTally::new(at, value));
    }

    let mut rows = Vec::with_capacity(tallies.len());
    for (label, tally) in tallies {
        let shape = tally.shape(label)?;
        rows.push(ObjectRow {
            label,
            shape: ShapeMeasurements::from_shape(&shape),
            intensity: tally.intensity(label, shape.at),
        });
    }
    Ok(rows)
}

fn rounded_centroid(position: [u64; 3], count: u64) -> Option<[usize; 3]> {
    if count == 0 {
        return None;
    }
    let count = count as u128;
    let mut out = [0usize; 3];
    for axis in 0..3 {
        let rounded = (2 * position[axis] as u128 + count) / (2 * count);
        out[axis] = usize::try_from(rounded).ok()?;
    }
    Some(out)
}

fn write_object_csv(rows: &[ObjectRow], path: &Path) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: create CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "label,count,centroid_z,centroid_y,centroid_x,bbox_min_z,bbox_min_y,bbox_min_x,\
         bbox_max_z,bbox_max_y,bbox_max_x,equivalent_radius,equivalent_diameter,\
         principal_axis_0,principal_axis_1,principal_axis_2,eccentricity,\
         intensity_count,finite_intensity_count,intensity_sum,intensity_mean,intensity_min,intensity_max"
    )
    .map_err(write_error)?;
    for row in rows {
        let centroid = row.shape.centroid.unwrap_or([f64::NAN; 3]);
        let axes = row.shape.principal_axis_lengths.unwrap_or([f64::NAN; 3]);
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.label,
            row.shape.count,
            centroid[0],
            centroid[1],
            centroid[2],
            row.shape.bbox_min[0],
            row.shape.bbox_min[1],
            row.shape.bbox_min[2],
            row.shape.bbox_max[0],
            row.shape.bbox_max[1],
            row.shape.bbox_max[2],
            row.shape.equivalent_sphere_radius,
            row.shape.equivalent_sphere_diameter,
            axes[0],
            axes[1],
            axes[2],
            optional(row.shape.eccentricity),
            row.intensity.count,
            row.intensity.finite_count,
            row.intensity.sum,
            optional(row.intensity.mean),
            optional(row.intensity.min),
            optional(row.intensity.max)
        )
        .map_err(write_error)?;
    }
    out.flush().map_err(write_error)
}

fn write_summary_json(
    config: &Config,
    rows: &[ObjectRow],
    threshold: f64,
    load_seconds: f64,
    pipeline_seconds: f64,
    path: &Path,
) -> Result<()> {
    let total_area: u64 = rows.iter().map(|row| row.shape.count).sum();
    let mean_area = if rows.is_empty() {
        0.0
    } else {
        total_area as f64 / rows.len() as f64
    };
    let text = format!(
        "{{\n\
         \"input\": {:?},\n\
         \"objects\": {},\n\
         \"total_foreground_area\": {},\n\
         \"mean_object_area\": {},\n\
         \"threshold\": {},\n\
         \"threshold_method\": {:?},\n\
         \"sigma\": {},\n\
         \"declump_sigma\": {},\n\
         \"min_size\": {},\n\
         \"max_size\": {},\n\
         \"seed_min_distance\": {},\n\
         \"maxima_downsample\": {},\n\
         \"declump_method\": {:?},\n\
         \"fill_holes_after_declumping\": {},\n\
         \"merge_line_basin_pixels\": {},\n\
         \"merge_line_max_saddle_drop\": {},\n\
         \"measurement_intensity_source\": \"input_luma_normalized_0_1\",\n\
         \"load_seconds\": {},\n\
         \"pipeline_seconds\": {}\n\
         }}\n",
        config.input.display().to_string(),
        rows.len(),
        total_area,
        mean_area,
        threshold,
        config.threshold_method.as_str(),
        config.sigma,
        config.declump_sigma,
        config.min_size,
        optional_u64_json(config.max_size),
        config.seed_min_distance,
        config.maxima_downsample,
        config.declump_method.as_str(),
        config.fill_holes_after_declumping,
        config.merge_line_basin_pixels,
        optional(config.merge_line_max_saddle_drop),
        load_seconds,
        pipeline_seconds
    );
    fs::write(path, text)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: write summary: {err}")))
}

fn save_mask(mask: &Array3<bool>, path: &Path) -> Result<()> {
    let shape = shape_of(mask);
    let mut image = ImageBuffer::<Luma<u8>, Vec<u8>>::new(shape[2] as u32, shape[1] as u32);
    for y in 0..shape[1] {
        for x in 0..shape[2] {
            image.put_pixel(
                x as u32,
                y as u32,
                Luma([if mask[[0, y, x]] { 255 } else { 0 }]),
            );
        }
    }
    image
        .save(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: save mask: {err}")))
}

fn save_labels(labels: &Array3<u32>, path: &Path) -> Result<()> {
    let shape = shape_of(labels);
    let mut image = ImageBuffer::<Luma<u16>, Vec<u16>>::new(shape[2] as u32, shape[1] as u32);
    for y in 0..shape[1] {
        for x in 0..shape[2] {
            let label = labels[[0, y, x]].min(u32::from(u16::MAX)) as u16;
            image.put_pixel(x as u32, y as u32, Luma([label]));
        }
    }
    image
        .save(path)
        .map_err(|err| Error::invalid(format!("cellprofiler-human: save labels: {err}")))
}

fn shape_of<T>(array: &Array3<T>) -> [usize; 3] {
    [array.shape()[0], array.shape()[1], array.shape()[2]]
}

fn view_shape<T>(array: &ndarray::ArrayView3<'_, T>) -> [usize; 3] {
    [array.shape()[0], array.shape()[1], array.shape()[2]]
}

fn optional(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NAN)
}

fn optional_u64_json(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| value.to_string())
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("cellprofiler-human: write output: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_declump_hole_filling_fills_label_enclosed_background() {
        let labels = Array3::from_shape_vec(
            (1, 5, 5),
            vec![
                0, 0, 0, 0, 0, //
                0, 7, 7, 7, 0, //
                0, 7, 0, 7, 0, //
                0, 7, 7, 7, 0, //
                0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        let filled = fill_label_holes_after_declumping(labels.view()).unwrap();
        assert_eq!(filled[[0, 2, 2]], 7);
    }

    #[test]
    fn post_declump_hole_filling_does_not_fill_background_connected_to_box_edge() {
        let labels = Array3::from_shape_vec(
            (1, 5, 5),
            vec![
                0, 0, 0, 0, 0, //
                0, 7, 0, 7, 0, //
                0, 7, 0, 7, 0, //
                0, 7, 7, 7, 0, //
                0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        let filled = fill_label_holes_after_declumping(labels.view()).unwrap();
        assert_eq!(filled[[0, 1, 2]], 0);
        assert_eq!(filled[[0, 2, 2]], 0);
    }

    #[test]
    fn maxima_downsample_one_matches_direct_regional_maxima() {
        let values = Array3::from_shape_vec(
            (1, 3, 3),
            vec![
                1.0, 2.0, 1.0, //
                2.0, 3.0, 2.0, //
                1.0, 2.0, 4.0,
            ],
        )
        .unwrap();
        let direct = regional_maxima(values.view()).unwrap();
        let downsampled = regional_maxima_for_seeding(values.view(), 1).unwrap();
        assert_eq!(downsampled, direct);
    }

    #[test]
    fn maxima_downsample_maps_low_resolution_maxima_to_best_source_pixel() {
        let values = Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                1.0, 9.0, 1.0, 4.0, //
                2.0, 8.0, 2.0, 2.0, //
                1.0, 1.0, 1.0, 3.0, //
                1.0, 1.0, 3.0, 2.0,
            ],
        )
        .unwrap();
        let maxima = regional_maxima_for_seeding(values.view(), 2).unwrap();
        assert!(maxima[[0, 0, 1]]);
        assert_eq!(maxima.iter().filter(|&&is_maximum| is_maximum).count(), 1);
    }

    #[test]
    fn line_basin_merge_joins_labels_with_enough_shared_line_pixels() {
        let labels = Array3::from_shape_vec(
            (1, 3, 5),
            vec![
                7, 7, 0, 9, 9, //
                7, 7, 0, 9, 9, //
                0, 0, 0, 5, 5,
            ],
        )
        .unwrap();
        let intensity = Array3::from_shape_vec(
            (1, 3, 5),
            vec![
                10.0, 10.0, 9.0, 10.0, 10.0, //
                10.0, 10.0, 9.0, 10.0, 10.0, //
                0.0, 0.0, 0.0, 3.0, 3.0,
            ],
        )
        .unwrap();
        let unchanged =
            merge_labels_across_watershed_lines(labels.view(), intensity.view(), 3, None).unwrap();
        assert_eq!(unchanged[[0, 0, 0]], 7);
        assert_eq!(unchanged[[0, 0, 3]], 9);

        let blocked =
            merge_labels_across_watershed_lines(labels.view(), intensity.view(), 2, Some(0.0))
                .unwrap();
        assert_eq!(blocked[[0, 0, 0]], 7);
        assert_eq!(blocked[[0, 0, 3]], 9);

        let merged =
            merge_labels_across_watershed_lines(labels.view(), intensity.view(), 2, Some(1.0))
                .unwrap();
        assert_eq!(merged[[0, 0, 0]], 7);
        assert_eq!(merged[[0, 0, 3]], 7);
        assert_eq!(merged[[0, 2, 3]], 5);
        assert_eq!(merged[[0, 0, 2]], 0);
    }
}
