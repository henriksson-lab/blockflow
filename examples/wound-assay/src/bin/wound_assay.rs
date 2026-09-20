// SPDX-License-Identifier: MIT

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use blockflow::assemble::{ImageId, PlanBuilder};
use blockflow::dtype::Dtype;
use blockflow::env::Environment;
use blockflow::geometry::BlockGrid;
use blockflow::op::Chain;
use blockflow::ops::VoxelwiseMaskOp;
use blockflow::region::Region;
use blockflow::strategy::{execute_phases, Hints};
use blockflow::voxels::Voxels;
use blockflow::zarr_env::ZarrEnvironment;
use blockflow::{AttachedImage, Error, Result};
use clap::Parser;
use ndarray::Array3;
use serde_json::json;

const HEIGHT: usize = 96;
const WIDTH: usize = 144;

#[derive(Debug, Parser)]
#[command(
    name = "wound-assay",
    about = "Measures scratch-assay images and reports open wound area."
)]
struct Config {
    #[arg(long, default_value = ".tmp/wound-assay/blockflow")]
    out: PathBuf,
    #[arg(long, default_value_t = 10)]
    images: usize,
    #[arg(long, default_value_t = 100.0)]
    threshold: f64,
    #[arg(long)]
    fixture_dir: Option<PathBuf>,
    #[arg(long, default_value = ".tmp/wound-assay/input.zarr")]
    zarr_dir: PathBuf,
    #[arg(long, value_parser = parse_chunk, default_value = "1x32x32")]
    chunk: [usize; 3],
}

#[derive(Debug)]
struct ImageRow {
    image: usize,
    open_area: u64,
    covered_area: u64,
    open_fraction: f64,
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
            "wound-assay: create output directory {}: {err}",
            config.out.display()
        ))
    })?;

    let mut rows = Vec::with_capacity(config.images);
    let mut profile = vec![0u64; WIDTH];
    let mut total_open = 0u64;
    for image in 0..config.images {
        let input = if let Some(dir) = &config.fixture_dir {
            image_from_fixture(dir, image)?
        } else {
            synthetic_image(image)
        };
        let input_zarr = ensure_wound_zarr(&config, image, input)?;
        let mask = planned_open_mask(&input_zarr, config.threshold, config.chunk)?;
        let (row, image_profile) = measure_mask(image, mask.view())?;
        total_open += row.open_area;
        for (dst, count) in profile.iter_mut().zip(image_profile) {
            *dst += count;
        }
        rows.push(row);
    }

    write_images(&rows, &config.out.join("images.csv"))?;
    write_profile(&profile, config.images, &config.out.join("profile.csv"))?;
    write_summary(&config, total_open, &config.out.join("summary.json"))?;

    println!(
        "images={} open_area={} output={}",
        config.images,
        total_open,
        config.out.display()
    );
    Ok(())
}

impl Config {
    fn parse() -> Result<Self> {
        let config = <Self as Parser>::parse();

        if config.images == 0 {
            return Err(Error::invalid("wound-assay: --images must be at least 1"));
        }
        if !config.threshold.is_finite() {
            return Err(Error::invalid("wound-assay: --threshold must be finite"));
        }
        if config.chunk.contains(&0) {
            return Err(Error::invalid(
                "wound-assay: --chunk dimensions must be positive",
            ));
        }

        Ok(config)
    }
}

struct WoundZarr {
    image: PathBuf,
    work: PathBuf,
}

fn ensure_wound_zarr(config: &Config, image: usize, input: Array3<f64>) -> Result<WoundZarr> {
    let root = config.zarr_dir.join(format!("image-{image:03}"));
    let store = root.join("image.zarr");
    let path = store.join("level0");
    if path.join("zarr.json").exists() {
        let (dtype, volume) = AttachedImage::at(&path).metadata()?;
        if dtype != Dtype::F64 || volume != [1, HEIGHT, WIDTH] {
            return Err(Error::invalid(format!(
                "wound-assay: prepared store {} is {dtype:?} {volume:?}, expected F64 {:?}",
                path.display(),
                [1, HEIGHT, WIDTH]
            )));
        }
    } else {
        let voxels: Voxels = input.into();
        ZarrEnvironment::create(&store, &voxels, config.chunk)?;
    }
    Ok(WoundZarr {
        image: path,
        work: root.join("work.zarr"),
    })
}

fn planned_open_mask(input: &WoundZarr, threshold: f64, chunk: [usize; 3]) -> Result<Array3<bool>> {
    let (_, volume) = AttachedImage::at(&input.image).metadata()?;
    let grid = BlockGrid::new(volume, chunk)?;
    let mut builder = PlanBuilder::new(volume, Dtype::F64, grid);
    builder.pixels(Chain::op(VoxelwiseMaskOp::new(
        "wound-open-mask",
        move |value| value < threshold,
    )))?;
    let base = builder.finish()?;
    let mask_image = ImageId::from(base.n_phases());
    let env = ZarrEnvironment::attach(&input.work, &[AttachedImage::at(&input.image)])?;
    let mut hints = Hints::default();
    hints.keep_images.insert(mask_image);
    execute_phases(
        "wound-assay planned open mask",
        &base.workflow,
        &base.decomposition,
        &hints,
        &env,
        &[],
        &base.work(),
    )?;
    let block = env.read(mask_image.index(), &Region::whole(&volume))?;
    block.as_array()?.view::<bool>().map(|view| view.to_owned())
}

fn synthetic_image(image: usize) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    for y in 0..HEIGHT {
        let (left, right) = wound_bounds(image, y);
        for x in 0..WIDTH {
            out[[0, y, x]] = if left <= x && x < right {
                42.0 + ((x + 3 * y + image) % 17) as f64
            } else {
                166.0 + ((2 * x + y + image) % 29) as f64
            };
        }
    }
    out
}

fn wound_bounds(image: usize, y: usize) -> (usize, usize) {
    let center = WIDTH as isize / 2 + (image % 5) as isize - 2;
    let half = 13 + (image % 4) as isize + ((y + image) % 9) as isize / 3;
    let drift = ((y * 7 + image * 3) % 11) as isize - 5;
    let left = center - half + drift / 2;
    let right = center + half + drift / 3;
    (
        left.max(0) as usize,
        right.clamp(0, WIDTH as isize) as usize,
    )
}

fn image_from_fixture(dir: &Path, image: usize) -> Result<Array3<f64>> {
    let path = dir.join(format!("wound-{image:03}.pgm"));
    let file = File::open(&path).map_err(|err| {
        Error::invalid(format!(
            "wound-assay: open fixture {}: {err}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::new(file);
    let magic = read_pgm_token(&mut reader, &path)?;
    if magic != "P5" {
        return Err(Error::invalid(format!(
            "wound-assay: fixture {} is not binary PGM P5",
            path.display()
        )));
    }
    let width: usize = parse_pgm_token(&mut reader, "width", &path)?;
    let height: usize = parse_pgm_token(&mut reader, "height", &path)?;
    let max_value: usize = parse_pgm_token(&mut reader, "max value", &path)?;
    if width != WIDTH || height != HEIGHT || max_value != 255 {
        return Err(Error::invalid(format!(
            "wound-assay: fixture {} has {width}x{height} max {max_value}, expected {WIDTH}x{HEIGHT} max 255",
            path.display()
        )));
    }
    let mut bytes = vec![0u8; WIDTH * HEIGHT];
    reader.read_exact(&mut bytes).map_err(|err| {
        Error::invalid(format!(
            "wound-assay: read fixture pixels {}: {err}",
            path.display()
        ))
    })?;
    let mut out = Array3::<f64>::zeros((1, HEIGHT, WIDTH));
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            out[[0, y, x]] = f64::from(bytes[y * WIDTH + x]);
        }
    }
    Ok(out)
}

fn read_pgm_token(reader: &mut BufReader<File>, path: &Path) -> Result<String> {
    let mut token = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        reader.read_exact(&mut byte).map_err(|err| {
            Error::invalid(format!(
                "wound-assay: read PGM token {}: {err}",
                path.display()
            ))
        })?;
        if byte[0] == b'#' {
            let mut discard = Vec::new();
            reader.read_until(b'\n', &mut discard).map_err(|err| {
                Error::invalid(format!(
                    "wound-assay: read PGM comment {}: {err}",
                    path.display()
                ))
            })?;
            continue;
        }
        if !byte[0].is_ascii_whitespace() {
            token.push(byte[0]);
            break;
        }
    }
    loop {
        reader.read_exact(&mut byte).map_err(|err| {
            Error::invalid(format!(
                "wound-assay: read PGM token {}: {err}",
                path.display()
            ))
        })?;
        if byte[0].is_ascii_whitespace() {
            break;
        }
        token.push(byte[0]);
    }
    String::from_utf8(token).map_err(|err| {
        Error::invalid(format!(
            "wound-assay: fixture {} contains non-UTF8 PGM token: {err}",
            path.display()
        ))
    })
}

fn parse_pgm_token<T: std::str::FromStr>(
    reader: &mut BufReader<File>,
    name: &str,
    path: &Path,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let raw = read_pgm_token(reader, path)?;
    raw.parse::<T>().map_err(|err| {
        Error::invalid(format!(
            "wound-assay: parse PGM {name}={raw:?} in {}: {err}",
            path.display()
        ))
    })
}

fn measure_mask(image: usize, mask: ndarray::ArrayView3<'_, bool>) -> Result<(ImageRow, Vec<u64>)> {
    if mask.shape() != [1, HEIGHT, WIDTH] {
        return Err(Error::invalid("wound-assay: unexpected mask shape"));
    }
    let mut open_area = 0u64;
    let mut profile = vec![0u64; WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if mask[[0, y, x]] {
                open_area += 1;
                profile[x] += 1;
            }
        }
    }
    let pixels = (HEIGHT * WIDTH) as u64;
    Ok((
        ImageRow {
            image,
            open_area,
            covered_area: pixels - open_area,
            open_fraction: open_area as f64 / pixels as f64,
        },
        profile,
    ))
}

fn write_images(rows: &[ImageRow], path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("wound-assay: create image CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "image,open_area,covered_area,open_fraction").map_err(write_error)?;
    for row in rows {
        writeln!(
            out,
            "{},{},{},{:.6}",
            row.image, row.open_area, row.covered_area, row.open_fraction
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_profile(profile: &[u64], images: usize, path: &PathBuf) -> Result<()> {
    let file = File::create(path)
        .map_err(|err| Error::invalid(format!("wound-assay: create profile CSV: {err}")))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "x,open_count,open_fraction").map_err(write_error)?;
    let denominator = (images * HEIGHT) as f64;
    for (x, count) in profile.iter().enumerate() {
        writeln!(out, "{x},{count},{:.6}", *count as f64 / denominator).map_err(write_error)?;
    }
    Ok(())
}

fn write_summary(config: &Config, open_area: u64, path: &PathBuf) -> Result<()> {
    let total = (config.images * HEIGHT * WIDTH) as u64;
    let summary = json!({
        "covered_area": total - open_area,
        "height": HEIGHT,
        "images": config.images,
        "input_zarr_dir": config.zarr_dir.display().to_string(),
        "open_area": open_area,
        "open_fraction": open_area as f64 / total as f64,
        "chunk_shape": config.chunk,
        "execution": "planned fixed-threshold open mask over attached Zarr inputs",
        "threshold": config.threshold,
        "width": WIDTH,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).expect("summary JSON must serialize") + "\n",
    )
    .map_err(|err| Error::invalid(format!("wound-assay: write summary JSON: {err}")))
}

fn write_error(err: std::io::Error) -> Error {
    Error::invalid(format!("wound-assay: write output: {err}"))
}

fn parse_chunk(raw: &str) -> std::result::Result<[usize; 3], String> {
    let parts = raw
        .split(['x', 'X', ',', ':'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!(
            "wound-assay: chunk shape {raw:?} must have three dimensions"
        ));
    }
    let mut out = [0usize; 3];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part
            .parse::<usize>()
            .map_err(|err| format!("wound-assay: could not parse chunk shape {raw:?}: {err}"))?;
    }
    Ok(out)
}
