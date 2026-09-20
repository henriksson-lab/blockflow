// SPDX-License-Identifier: MIT

#[path = "../../../support/planning.rs"]
mod example_planning;

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use blockflow::assemble::ImageId;
use blockflow::dtype::Dtype;
use blockflow::op::Chain;
use blockflow::ops::measure::{
    collect_class_a_shapes, collect_class_a_values, IntensityImage, IntensityMeasurements,
    IntensitySet, Measurements, ShapeMeasurements, ShapeSet,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints, Workflow};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Error, Result};
use clap::Parser;
use ndarray::Array3;
use serde_json::json;

const HEIGHT: usize = 104;
const WIDTH: usize = 136;
const NUCLEI_PER_IMAGE: usize = 5;

#[derive(Debug, Parser)]
#[command(
    name = "foci-per-nucleus",
    about = "Generates deterministic nucleus labels and foci, then counts foci by containing nucleus."
)]
struct Config {
    #[arg(long, default_value = ".tmp/foci-per-nucleus/blockflow")]
    out: PathBuf,
    #[arg(long, default_value_t = 10)]
    images: usize,
    #[arg(long, default_value = ".tmp/foci-per-nucleus/input.zarr")]
    zarr_dir: PathBuf,
    #[arg(long, value_parser = parse_chunk, default_value = "1x32x32")]
    chunk: [usize; 3],
    #[arg(long, default_value_t = false)]
    prepare_only: bool,
}

#[derive(Debug)]
struct NucleusRow {
    image: usize,
    label: u64,
    area: u64,
    foci_count: u64,
    foci_intensity_sum: f64,
}

#[derive(Debug)]
struct FocusRow {
    image: usize,
    focus: u64,
    y: usize,
    x: usize,
    intensity: f64,
    nucleus_label: u64,
}

#[derive(Debug, Clone, Copy)]
struct FocusSpec {
    y: usize,
    x: usize,
    intensity: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = Config::parse()?;
    fs::create_dir_all(&config.out).map_err(|err| {
        Error::invalid(format!(
            "foci-per-nucleus: create output directory {}: {err}",
            config.out.display()
        ))
    })?;

    let mut nuclei = Vec::new();
    let mut foci = Vec::new();
    for image in 0..config.images {
        let labels = synthetic_nuclei(image);
        let specs = synthetic_foci(image);
        let image_foci = focus_rows(image, labels.view(), &specs);
        let root = config.zarr_dir.join(format!("image-{image:03}"));
        let labels_path = root.join("labels.zarr/level0");
        let counts_path = root.join("focus-counts.zarr/level0");
        let intensities_path = root.join("focus-intensities.zarr/level0");
        let input = if [&labels_path, &counts_path, &intensities_path]
            .iter()
            .all(|path| path.join("zarr.json").exists())
        {
            FociZarr {
                labels: labels_path,
                counts: counts_path,
                intensities: intensities_path,
            }
        } else {
            let (focus_counts, focus_intensities) = foci_images(&specs);
            ensure_foci_zarr(&config, image, labels, focus_counts, focus_intensities)?
        };
        if config.prepare_only {
            continue;
        }
        let mut image_nuclei = planned_nucleus_foci(image, &input, &config.out)?;
        nuclei.append(&mut image_nuclei);
        foci.extend(image_foci);
    }
    if config.prepare_only {
        return Ok(());
    }
    nuclei.sort_by_key(|row| (row.image, row.label));
    foci.sort_by_key(|row| (row.image, row.focus));

    write_nuclei(&nuclei, &config.out.join("nuclei.csv"))?;
    write_foci(&foci, &config.out.join("foci.csv"))?;
    write_summary(&config, &nuclei, &foci, &config.out.join("summary.json"))?;

    let assigned = foci.iter().filter(|row| row.nucleus_label != 0).count();
    println!(
        "images={} nuclei={} foci={} assigned={} output={}",
        config.images,
        nuclei.len(),
        foci.len(),
        assigned,
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let config = <Self as Parser>::parse();

        if config.images == 0 {
            return Err(Error::invalid(
                "foci-per-nucleus: --images must be at least 1",
            ));
        }
        if config.chunk.contains(&0) {
            return Err(Error::invalid(
                "foci-per-nucleus: --chunk dimensions must be positive",
            ));
        }

        Ok(config)
    }
}

struct FociZarr {
    labels: PathBuf,
    counts: PathBuf,
    intensities: PathBuf,
}

fn ensure_foci_zarr(
    config: &Config,
    image: usize,
    labels: Array3<u32>,
    counts: Array3<f64>,
    intensities: Array3<f64>,
) -> Result<FociZarr> {
    let root = config.zarr_dir.join(format!("image-{image:03}"));
    let labels = ensure_array_zarr(&root.join("labels.zarr"), labels, config.chunk)?;
    let counts = ensure_array_zarr(&root.join("focus-counts.zarr"), counts, config.chunk)?;
    let intensities = ensure_array_zarr(
        &root.join("focus-intensities.zarr"),
        intensities,
        config.chunk,
    )?;
    Ok(FociZarr {
        labels,
        counts,
        intensities,
    })
}

fn ensure_array_zarr<T>(store: &Path, array: Array3<T>, chunk: [usize; 3]) -> Result<PathBuf>
where
    T: blockflow::voxels::VoxelElement + 'static,
    Voxels: From<Array3<T>>,
{
    let path = store.join("level0");
    if path.join("zarr.json").exists() {
        let (_, volume) = AttachedImage::at(&path).metadata()?;
        if volume != [1, HEIGHT, WIDTH] {
            return Err(Error::invalid(format!(
                "foci-per-nucleus: prepared store {} is volume {volume:?}, expected {:?}",
                path.display(),
                [1, HEIGHT, WIDTH]
            )));
        }
        return Ok(path);
    }

    let voxels: Voxels = array.into();
    ZarrEnvironment::create(store, &voxels, chunk)?;
    Ok(path)
}

fn planned_nucleus_foci(
    image: usize,
    input: &FociZarr,
    output_dir: &Path,
) -> Result<Vec<NucleusRow>> {
    let images = [
        AttachedImage::at(&input.labels),
        AttachedImage::at(&input.counts),
        AttachedImage::at(&input.intensities),
    ];
    let (dtype, volume) = images[0].metadata()?;
    let measurements = Measurements::for_labels(ImageId::from(0))
        .shape(ShapeSet::basic())
        .intensity(
            IntensityImage::<0>::new(ImageId::supplied(0)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .intensity(
            IntensityImage::<1>::new(ImageId::supplied(1)).holding(Dtype::F64),
            IntensitySet::standard(),
        )
        .stream("foci-per-nucleus.measurements")
        .lifecycle(Lifecycle::DeleteOnExit)
        .plan_on_attached(&images, &example_planning::constraints(volume))?;
    let shape_rows = measurements
        .class_a_rows()
        .ok_or_else(|| Error::invalid("foci-per-nucleus: planned shape rows are missing"))?;
    let count_rows = measurements
        .class_a_intensity_rows(0)
        .ok_or_else(|| Error::invalid("foci-per-nucleus: planned focus-count rows are missing"))?;
    let intensity_rows = measurements.class_a_intensity_rows(1).ok_or_else(|| {
        Error::invalid("foci-per-nucleus: planned focus-intensity rows are missing")
    })?;
    let scratch = tempfile::tempdir_in(output_dir).map_err(Error::backend)?;
    let env = ZarrEnvironment::attach(scratch.path(), &images)?;
    let workflow = Workflow::new(Chain::sequence(Vec::new()), volume, dtype);
    let work = measurements.phase_work();
    execute_phases(
        "foci-per-nucleus planned measurement",
        &workflow,
        &measurements.decomposition,
        &Hints::default(),
        &env,
        &[],
        &work,
    )?;

    let shapes = collect_class_a_shapes(&env, &shape_rows, volume, measurements.fixed)?;
    let mut counts = collect_class_a_values(&env, &count_rows, volume, measurements.fixed)?
        .into_iter()
        .map(|values| {
            let measurement = IntensityMeasurements::from_values(&values);
            (measurement.label, measurement)
        })
        .collect::<BTreeMap<_, _>>();
    let mut intensities =
        collect_class_a_values(&env, &intensity_rows, volume, measurements.fixed)?
            .into_iter()
            .map(|values| {
                let measurement = IntensityMeasurements::from_values(&values);
                (measurement.label, measurement)
            })
            .collect::<BTreeMap<_, _>>();

    let mut rows = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let shape = ShapeMeasurements::from_shape(&shape);
        let count = counts.remove(&shape.label).ok_or_else(|| {
            Error::invalid(format!(
                "foci-per-nucleus: no planned focus count for label {}",
                shape.label
            ))
        })?;
        let intensity = intensities.remove(&shape.label).ok_or_else(|| {
            Error::invalid(format!(
                "foci-per-nucleus: no planned focus intensity for label {}",
                shape.label
            ))
        })?;
        rows.push(NucleusRow {
            image,
            label: shape.label,
            area: shape.count,
            foci_count: count.sum.round() as u64,
            foci_intensity_sum: intensity.sum,
        });
    }
    Ok(rows)
}

fn synthetic_nuclei(image: usize) -> Array3<u32> {
    let mut labels = Array3::<u32>::zeros((1, HEIGHT, WIDTH));
    for local in 0..NUCLEI_PER_IMAGE {
        let label = (image * 100 + local + 1) as u32;
        let (y0, x0, height, width) = nucleus_rect(image, local);
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                labels[[0, y, x]] = label;
            }
        }
    }
    labels
}

fn nucleus_rect(image: usize, local: usize) -> (usize, usize, usize, usize) {
    let y0 = 12 + (local / 3) * 42 + (image % 4);
    let x0 = 10 + (local % 3) * 40 + ((image + local) % 6);
    let height = 24 + ((image + 2 * local) % 5);
    let width = 26 + ((2 * image + local) % 7);
    (y0, x0, height, width)
}

fn synthetic_foci(image: usize) -> Vec<FocusSpec> {
    let mut specs = Vec::new();
    for local in 0..NUCLEI_PER_IMAGE {
        let (y0, x0, height, width) = nucleus_rect(image, local);
        let count = 1 + ((image + local) % 3);
        for index in 0..count {
            let y = y0 + 3 + ((image + 5 * index + local) % (height - 6).max(1));
            let x = x0 + 4 + ((2 * image + 7 * index + local) % (width - 8).max(1));
            let intensity = 180.0 + 11.0 * index as f64 + 3.0 * local as f64 + (image % 5) as f64;
            specs.push(FocusSpec { y, x, intensity });
        }
    }
    specs.push(FocusSpec {
        y: 2 + image % 5,
        x: 3 + image % 7,
        intensity: 99.0,
    });
    let (y0, x0, _height, width) = nucleus_rect(image, 0);
    specs.push(FocusSpec {
        y: y0,
        x: x0 + width,
        intensity: 120.0,
    });
    specs
}

fn foci_images(specs: &[FocusSpec]) -> (Array3<f64>, Array3<f64>) {
    let mut counts = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    let mut intensities = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    for spec in specs {
        counts[[0, spec.y, spec.x]] += 1.0;
        intensities[[0, spec.y, spec.x]] += spec.intensity;
    }
    (counts, intensities)
}

fn focus_rows(
    image: usize,
    labels: ndarray::ArrayView3<'_, u32>,
    specs: &[FocusSpec],
) -> Vec<FocusRow> {
    let mut foci = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let label = u64::from(labels[[0, spec.y, spec.x]]);
        foci.push(FocusRow {
            image,
            focus: (image * 1000 + index + 1) as u64,
            y: spec.y,
            x: spec.x,
            intensity: spec.intensity,
            nucleus_label: label,
        });
    }
    foci
}

fn write_nuclei(rows: &[NucleusRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("foci-per-nucleus: create nuclei CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "image,label,area,foci_count,foci_intensity_sum").map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{},{:.6}",
            row.image, row.label, row.area, row.foci_count, row.foci_intensity_sum
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_foci(rows: &[FocusRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("foci-per-nucleus: create foci CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "image,focus,y,x,intensity,nucleus_label").map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{},{:.6},{}",
            row.image, row.focus, row.y, row.x, row.intensity, row.nucleus_label
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_summary(
    config: &Config,
    nuclei: &[NucleusRow],
    foci: &[FocusRow],
    path: &PathBuf,
) -> Result<()> {
    let assigned = foci.iter().filter(|row| row.nucleus_label != 0).count();
    let summary = json!({
        "assigned_foci": assigned,
        "chunk_shape": config.chunk,
        "execution": "planned nucleus shape and foci intensity reductions over attached Zarr inputs",
        "images": config.images,
        "input_zarr_dir": config.zarr_dir.display().to_string(),
        "nuclei": nuclei.len(),
        "total_foci": foci.len(),
        "unassigned_foci": foci.len() - assigned,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).expect("summary JSON must serialize") + "\n",
    )
    .map_err(|err| Error::invalid(format!("foci-per-nucleus: write summary JSON: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("foci-per-nucleus: write output: {err}"))
}

fn parse_chunk(raw: &str) -> std::result::Result<[usize; 3], String> {
    let parts = raw
        .split(['x', 'X', ',', ':'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "foci-per-nucleus: chunk shape {raw:?} must have three dimensions"
        ));
    }
    let mut out = [0usize; 3];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part.parse::<usize>().map_err(|err| {
            format!("foci-per-nucleus: could not parse chunk shape {raw:?}: {err}")
        })?;
    }
    Ok(out)
}
