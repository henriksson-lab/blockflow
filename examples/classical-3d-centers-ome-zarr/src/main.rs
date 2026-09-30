use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use blockflow::classical_centers::{
    append_classical_center_phases, center_schema, finalize_centers, finalize_watershed_instances,
    CenterSeedsOp, CenterTable, ClassicalCenterConfig, ClassicalCenterResponse,
    WatershedInstanceTable,
};
use blockflow::label_pyramid::{build_nearest_label_pyramid_with_blocks, refresh_label_registry};
use blockflow::ome_zarr::{write_label_metadata, OmeVolumePyramid};
use blockflow::ops::{
    append_bounded_otsu_threshold_phases, BlobResponse, Boundary, FixedPoint, Gaussian,
    GlobalThresholdOutput, MergeTabulationOp, SeededWatershedOp, Separation, SmoothOp,
    TabulateValuesOp, ThresholdTest, VoxelwiseMapOp,
};
use blockflow::sidecar::Lifecycle;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::{
    AttachedImage, BlockGrid, Chain, Dtype, Error, ImageId, PlanBuilder, Result, ZarrEnvironment,
};
use clap::{Parser, ValueEnum};

const ROWS: &str = "classical-centers";

#[derive(Debug, Parser)]
#[command(about = "Detect bright 3D cell centres in an OME-Zarr channel")]
struct Args {
    #[arg(long)]
    zarr: PathBuf,
    #[arg(long, default_value_t = 0)]
    channel: usize,
    #[arg(long, default_value_t = 0)]
    time: usize,
    #[arg(long, value_enum, default_value_t = Method::Dog)]
    method: Method,
    /// Narrow Gaussian sigma in physical units.
    #[arg(long, default_value_t = 2.0)]
    sigma_um: f64,
    /// Wide/narrow sigma ratio for Difference of Gaussians.
    #[arg(long, default_value_t = 1.6)]
    dog_ratio: f64,
    #[arg(long, default_value_t = 3.0)]
    truncate: f64,
    #[arg(long, default_value_t = 256)]
    histogram_bins: usize,
    #[arg(long, default_value_t = 4.0)]
    min_distance_um: f64,
    #[arg(long, default_value_t = 32)]
    block_z: usize,
    #[arg(long, default_value_t = 256)]
    block: usize,
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value = "classical-3d-centers-v1")]
    layer: String,
    #[arg(long, default_value_t = false)]
    overwrite: bool,
    /// Also produce a foreground-constrained instance watershed.
    #[arg(long, default_value_t = false)]
    watershed: bool,
    /// Optional level-0 crop origin as z,y,x. Center coordinates remain global.
    #[arg(long, value_delimiter = ',', num_args = 3, requires = "crop_shape")]
    crop_start: Option<Vec<usize>>,
    /// Optional level-0 crop shape as z,y,x.
    #[arg(long, value_delimiter = ',', num_args = 3, requires = "crop_start")]
    crop_shape: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Method {
    Gaussian,
    Dog,
    Log,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::invalid("this example requires cargo run --release"));
    }
    let args = Args::parse();
    validate(&args)?;
    let started = Instant::now();
    let pyramid = OmeVolumePyramid::open(&args.zarr, args.channel, args.time)?;
    let voxel_size = pyramid.voxel_size();
    let sigma = voxel_size.map(|spacing| args.sigma_um / spacing);
    let response = match args.method {
        Method::Gaussian => {
            ClassicalCenterResponse::gaussian(sigma, args.truncate, Boundary::Reflect)?
        }
        Method::Dog => ClassicalCenterResponse::Blob(BlobResponse::difference_of_gaussians(
            sigma,
            sigma.map(|value| value * args.dog_ratio),
            args.truncate,
            Boundary::Reflect,
        )?),
        Method::Log => ClassicalCenterResponse::Blob(BlobResponse::laplacian_of_gaussian(
            sigma,
            args.truncate,
            Boundary::Reflect,
        )?),
    };
    let index_volume = pyramid
        .levels()
        .first()
        .ok_or_else(|| Error::invalid("OME-Zarr image has no multiscale levels"))?
        .shape;
    let origin = triple(args.crop_start.as_deref()).unwrap_or([0; 3]);
    let crop_shape = triple(args.crop_shape.as_deref());
    if args.watershed && crop_shape.is_some() {
        return Err(Error::invalid(
            "--watershed with --crop-shape needs translated label metadata; use a cropped OME-Zarr fixture",
        ));
    }
    let source = match crop_shape {
        Some(shape) => pyramid.attached_level(0)?.window(origin, shape),
        None => pyramid.attached_level(0)?,
    };
    let (dtype, volume) = source.metadata()?;
    let grid = BlockGrid::new(volume, [args.block_z, args.block, args.block])?;
    let mut builder = PlanBuilder::new(volume, dtype, grid);
    let (rows_phase, _schema) = append_classical_center_phases(
        &mut builder,
        "classical-center-candidates",
        ROWS,
        ClassicalCenterConfig {
            response,
            histogram_bins: args.histogram_bins,
            minimum_distance_physical: args.min_distance_um,
            voxel_size,
        },
    )?;
    let watershed_phase = if args.watershed {
        let mask = append_bounded_otsu_threshold_phases(
            &mut builder,
            ImageId::from(0usize),
            "classical-watershed-foreground",
            args.histogram_bins,
            GlobalThresholdOutput::Mask {
                test: ThresholdTest::AtOrAbove,
            },
        )?
        .image()?;
        let seeds = builder
            .fragments(CenterSeedsOp::new(
                "classical centre seeds",
                ROWS,
                rows_phase.index(),
                center_schema()?,
            ))?
            .image()?;

        // The exact flood has whole-volume reach. One whole-volume block avoids
        // recomputing the same global watershed once per storage block.
        builder.regrid(BlockGrid::new(volume, volume)?);
        let gaussian = Gaussian::new(sigma, args.truncate)?.with_boundary(Boundary::Reflect);
        builder.pixels(Chain::sequence(vec![
            Chain::source(ImageId::from(0usize), dtype),
            Chain::op(SmoothOp::new("classical watershed smoothing", gaussian)),
            Chain::op(VoxelwiseMapOp::new(
                "classical watershed negative intensity",
                |value: f64| -value,
            )),
        ]))?;
        let labels = builder.pixels(Chain::op(
            SeededWatershedOp::new("classical centre watershed", seeds, Separation::Line)
                .within(mask),
        ))?;
        Some(labels)
    } else {
        None
    };
    let assembly = builder.finish()?;

    let work = args
        .zarr
        .join("tables")
        .join(format!(".{}-blockflow-work", args.layer));
    let center_layer = if args.watershed {
        format!("{}-centers", args.layer)
    } else {
        args.layer.clone()
    };
    let center_destination = args.zarr.join("tables").join(center_layer);
    let label_destination = args.zarr.join("labels").join(&args.layer);
    let instance_destination = args.zarr.join("tables").join(&args.layer);
    check_destination(&center_destination, args.overwrite)?;
    if args.watershed {
        check_destination(&label_destination, args.overwrite)?;
        check_destination(&instance_destination, args.overwrite)?;
    }
    if work.exists() {
        fs::remove_dir_all(&work).map_err(Error::backend)?;
    }
    let env = ZarrEnvironment::attach(&work, std::slice::from_ref(&source))?;
    execute_phases(
        "classical 3D centres OME-Zarr",
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
    let staged = work.join("center-table");
    let table = finalize_centers(
        &env,
        &staged,
        &work,
        CenterTable {
            stream: ROWS,
            phase: rows_phase.index(),
            volume,
            origin,
            index_volume,
        },
    )?;
    let detections = table.spec().row_count;
    write_csv(&table, &staged.join("table.csv"))?;
    drop(table);
    drop(env);
    let watershed_objects = if let Some(labels) = watershed_phase {
        let staged_labels = work.join("labels");
        fs::create_dir_all(&staged_labels).map_err(Error::backend)?;
        fs::rename(
            work.join(format!("level{}", labels.image()?.index())),
            staged_labels.join("0"),
        )
        .map_err(Error::backend)?;

        let measurement_work = work.join("measurements");
        let measurement_grid = BlockGrid::new(volume, [args.block_z, args.block, args.block])?;
        let mut measurements = PlanBuilder::new(volume, Dtype::U32, measurement_grid);
        let fixed = FixedPoint::bits(0)?;
        let partials = measurements.fragments(
            TabulateValuesOp::new(
                "classical watershed measurements",
                ImageId::from(0usize),
                ImageId::supplied(0),
                fixed,
                "classical-watershed-partials",
                Lifecycle::DeleteOnExit,
            )?
            .holding(Dtype::U32, dtype),
        )?;
        let rows = measurements.fragments(MergeTabulationOp::new(
            "classical watershed measurement merge",
            "classical-watershed-partials",
            partials.index(),
            measurements.grid().blocks_per_axis(),
            fixed,
            "classical-watershed-instances",
            Lifecycle::Persistent,
        ))?;
        let measurement_assembly = measurements.finish()?;
        let label_source = AttachedImage::at(staged_labels.join("0"));
        let measurement_env =
            ZarrEnvironment::attach(&measurement_work, &[label_source, source.clone()])?;
        execute_phases(
            "classical 3D watershed measurements",
            &measurement_assembly.workflow,
            &measurement_assembly.decomposition,
            &Hints {
                concurrency: args.workers.max(1),
                ..Hints::default()
            },
            &measurement_env,
            &[],
            &measurement_assembly.work(),
        )?;
        let staged_instances = work.join("instance-table");
        let instances = finalize_watershed_instances(
            &measurement_env,
            &staged_instances,
            &work,
            WatershedInstanceTable {
                stream: "classical-watershed-instances",
                phase: rows.index(),
                volume,
                fixed,
                layer: &args.layer,
                volume_per_voxel: pyramid.voxel_volume(),
            },
        )?;
        let count = instances.spec().row_count;
        drop(instances);
        drop(measurement_env);
        Some((staged_labels, staged_instances, count))
    } else {
        None
    };
    replace_with(&staged, &center_destination, args.overwrite)?;
    if let Some((staged_labels, staged_instances, count)) = watershed_objects {
        let shapes = pyramid
            .levels()
            .iter()
            .map(|level| level.shape)
            .collect::<Vec<_>>();
        build_nearest_label_pyramid_with_blocks(
            &staged_labels,
            &work.join("pyramid-work"),
            &shapes,
            &[[args.block_z, args.block, args.block]],
            args.workers,
        )?;
        write_label_metadata(&staged_labels, pyramid.levels(), pyramid.label_source())?;
        replace_with(&staged_labels, &label_destination, args.overwrite)?;
        replace_with(&staged_instances, &instance_destination, args.overwrite)?;
        refresh_label_registry(&args.zarr.join("labels"))?;
        println!(
            "watershed_objects={} label={} table={}",
            count,
            label_destination.display(),
            instance_destination.display()
        );
    }
    fs::remove_dir_all(&work).map_err(Error::backend)?;
    println!(
        "detections={} elapsed_seconds={:.3} table={}",
        detections,
        started.elapsed().as_secs_f64(),
        center_destination.display()
    );
    Ok(())
}

fn validate(args: &Args) -> Result<()> {
    if args.block_z == 0
        || args.block == 0
        || args.histogram_bins == 0
        || !args.sigma_um.is_finite()
        || args.sigma_um <= 0.0
        || !args.dog_ratio.is_finite()
        || args.dog_ratio <= 1.0
        || !args.truncate.is_finite()
        || args.truncate <= 0.0
        || !args.min_distance_um.is_finite()
        || args.min_distance_um < 0.0
    {
        return Err(Error::invalid("invalid classical centre parameters"));
    }
    if args.layer.is_empty()
        || args.layer.contains('/')
        || args.layer.contains('\\')
        || args.layer == "."
        || args.layer == ".."
    {
        return Err(Error::invalid("--layer must be one path component"));
    }
    Ok(())
}

fn check_destination(path: &Path, overwrite: bool) -> Result<()> {
    if path.exists() && !overwrite {
        return Err(Error::invalid(format!(
            "{} exists; pass --overwrite to replace it",
            path.display()
        )));
    }
    Ok(())
}

fn replace_with(staged: &Path, destination: &Path, overwrite: bool) -> Result<()> {
    if destination.exists() {
        if !overwrite {
            return Err(Error::invalid(format!(
                "{} appeared during the run and will not be replaced",
                destination.display()
            )));
        }
        fs::remove_dir_all(destination).map_err(Error::backend)?;
    }
    fs::create_dir_all(destination.parent().expect("destination has a parent"))
        .map_err(Error::backend)?;
    fs::rename(staged, destination).map_err(Error::backend)
}

fn triple(values: Option<&[usize]>) -> Option<[usize; 3]> {
    values.map(|values| values.try_into().expect("clap requires exactly z,y,x"))
}

fn write_csv(table: &ngff_object_table::TableReader, path: &Path) -> Result<()> {
    let range = 0..table.spec().row_count;
    let ids = table
        .read_u64("detection_id", range.clone())
        .map_err(Error::backend)?;
    let names = [
        "z",
        "y",
        "x",
        "response",
        "raw_intensity",
        "smoothed_intensity",
        "scale_z",
        "scale_y",
        "scale_x",
    ];
    let columns = names
        .iter()
        .map(|name| table.read_f32(name, range.clone()).map_err(Error::backend))
        .collect::<Result<Vec<_>>>()?;
    let mut writer = BufWriter::new(fs::File::create(path).map_err(Error::backend)?);
    writeln!(writer, "detection_id,{}", names.join(",")).map_err(Error::backend)?;
    for row in 0..ids.len() {
        write!(writer, "{}", ids[row]).map_err(Error::backend)?;
        for column in &columns {
            write!(writer, ",{}", column[row]).map_err(Error::backend)?;
        }
        writeln!(writer).map_err(Error::backend)?;
    }
    writer.flush().map_err(Error::backend)
}
