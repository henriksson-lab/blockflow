use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use ngff_object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType, SpatialIndexSpec, TableReader, TableSpec,
    TableWriter,
};
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Deserialize)]
struct CsvRow {
    label_id: u64,
    area_pixels: u32,
    area_um2: f64,
    centroid_y: f64,
    centroid_x: f64,
    dapi_mean: f64,
    dapi_min: u8,
    dapi_max: u8,
}

fn main() -> Result<()> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() == 2 && arguments[0] == "--verify" {
        return verify(Path::new(&arguments[1]));
    }
    if arguments.len() != 5 {
        return Err(format!(
            "usage: ngff-object-table-convert INPUT.csv OUTPUT.zarr LABEL HEIGHT WIDTH\n       \
             ngff-object-table-convert --verify TABLE.zarr"
        )
        .into());
    }
    let input = PathBuf::from(&arguments[0]);
    let output = PathBuf::from(&arguments[1]);
    let label = &arguments[2];
    let height = arguments[3].parse::<u64>()?;
    let width = arguments[4].parse::<u64>()?;
    convert(&input, &output, label, height, width)
}

fn verify(root: &Path) -> Result<()> {
    let reader = TableReader::open(root)?;
    let rows = reader.spec().row_count;
    let physical_ids = reader.read_u64(&reader.spec().identity, 0..rows)?;
    let (ids, physical_rows) = reader.read_identity_index(0..rows)?;
    if ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("identity index is not strictly increasing".into());
    }
    for (&id, &physical_row) in ids.iter().zip(&physical_rows) {
        let Some(&stored) = physical_ids.get(physical_row as usize) else {
            return Err(format!("identity {id} points outside the table").into());
        };
        if stored != id {
            return Err(format!(
                "identity index maps {id} to row {physical_row}, which stores {stored}"
            )
            .into());
        }
    }
    let (_, counts) = reader.read_spatial_index()?;
    if counts.iter().sum::<u64>() != rows {
        return Err("spatial index does not account for every row".into());
    }
    let level = reader.occupancy_shapes().len() - 1;
    let shape = reader.occupancy_shapes()[level].clone();
    let occupancy = reader.read_occupancy(level, &vec![0; shape.len()], &shape)?;
    if occupancy.iter().sum::<u64>() != rows {
        return Err("occupancy pyramid does not account for every row".into());
    }
    println!("verified {rows} rows at {}", root.display());
    Ok(())
}

fn convert(input: &Path, output: &Path, label: &str, height: u64, width: u64) -> Result<()> {
    const TILE: u64 = 1024;
    const CHUNK: u64 = 2048;

    let rows = csv::Reader::from_path(input)?
        .deserialize::<CsvRow>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let csv_rows = rows.len();
    let mut rows = merge_duplicate_ids(rows)?;
    for row in &rows {
        if !row.centroid_y.is_finite()
            || !row.centroid_x.is_finite()
            || row.centroid_y < 0.0
            || row.centroid_x < 0.0
            || row.centroid_y >= height as f64
            || row.centroid_x >= width as f64
            || !row.area_um2.is_finite()
            || !row.dapi_mean.is_finite()
        {
            return Err(format!("row for label {} has an invalid value", row.label_id).into());
        }
    }
    let grid_shape = vec![height.div_ceil(TILE), width.div_ceil(TILE)];
    rows.sort_unstable_by(|left, right| {
        row_key(left, &grid_shape).cmp(&row_key(right, &grid_shape))
    });

    let columns = vec![
        ColumnSpec::new("label_id", DType::U64, ColumnRole::Identity),
        ColumnSpec::new("centroid_y", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("centroid_x", DType::F32, ColumnRole::Coordinate),
        ColumnSpec::new("area_pixels", DType::U32, ColumnRole::Measurement),
        ColumnSpec::new("area_um2", DType::F32, ColumnRole::Measurement),
        ColumnSpec::new("dapi_mean", DType::F32, ColumnRole::Intensity),
        ColumnSpec::new("dapi_min", DType::U8, ColumnRole::Intensity),
        ColumnSpec::new("dapi_max", DType::U8, ColumnRole::Intensity),
    ];
    let spatial_index = SpatialIndexSpec {
        coordinates: vec![
            CoordinateColumn {
                axis: "y".into(),
                column: "centroid_y".into(),
            },
            CoordinateColumn {
                axis: "x".into(),
                column: "centroid_x".into(),
            },
        ],
        tile_shape: vec![TILE, TILE],
        grid_shape: grid_shape.clone(),
        tile_order: "row_major".into(),
        within_tile_order: "lexicographic_coordinates_then_identity".into(),
    };
    let mut spec = TableSpec::new(
        rows.len() as u64,
        CHUNK,
        "../../",
        "label_id",
        columns,
        spatial_index,
    );
    spec.region = Some(format!("../../labels/{label}"));
    let writer = TableWriter::create(output, spec)?;

    for (chunk_index, chunk) in rows.chunks(CHUNK as usize).enumerate() {
        let start = chunk_index as u64 * CHUNK;
        writer.write_u64(
            "label_id",
            start,
            &chunk.iter().map(|row| row.label_id).collect::<Vec<_>>(),
        )?;
        writer.write_f32(
            "centroid_y",
            start,
            &chunk
                .iter()
                .map(|row| row.centroid_y as f32)
                .collect::<Vec<_>>(),
        )?;
        writer.write_f32(
            "centroid_x",
            start,
            &chunk
                .iter()
                .map(|row| row.centroid_x as f32)
                .collect::<Vec<_>>(),
        )?;
        writer.write_u32(
            "area_pixels",
            start,
            &chunk.iter().map(|row| row.area_pixels).collect::<Vec<_>>(),
        )?;
        writer.write_f32(
            "area_um2",
            start,
            &chunk
                .iter()
                .map(|row| row.area_um2 as f32)
                .collect::<Vec<_>>(),
        )?;
        writer.write_f32(
            "dapi_mean",
            start,
            &chunk
                .iter()
                .map(|row| row.dapi_mean as f32)
                .collect::<Vec<_>>(),
        )?;
        writer.write_u8(
            "dapi_min",
            start,
            &chunk.iter().map(|row| row.dapi_min).collect::<Vec<_>>(),
        )?;
        writer.write_u8(
            "dapi_max",
            start,
            &chunk.iter().map(|row| row.dapi_max).collect::<Vec<_>>(),
        )?;
    }

    let mut identities = rows
        .iter()
        .enumerate()
        .map(|(physical_row, row)| (row.label_id, physical_row as u64))
        .collect::<Vec<_>>();
    identities.sort_unstable();
    if identities.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("CSV contains duplicate label_id values".into());
    }
    for (chunk_index, chunk) in identities.chunks(CHUNK as usize).enumerate() {
        writer.write_identity_index(
            chunk_index as u64 * CHUNK,
            &chunk.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            &chunk.iter().map(|pair| pair.1).collect::<Vec<_>>(),
        )?;
    }

    let tile_count = grid_shape.iter().product::<u64>() as usize;
    let mut counts = vec![0u64; tile_count];
    for row in &rows {
        counts[tile_id(row, &grid_shape)] += 1;
    }
    let mut starts = vec![0u64; tile_count];
    let mut next = 0;
    for (start, count) in starts.iter_mut().zip(&counts) {
        *start = next;
        next += count;
    }
    writer.write_spatial_index(&starts, &counts)?;
    let reader = writer.finish()?;
    let top = reader.occupancy_shapes().len() - 1;
    let top_shape = reader.occupancy_shapes()[top].clone();
    let occupied = reader.read_occupancy(top, &vec![0; top_shape.len()], &top_shape)?;
    if occupied.iter().sum::<u64>() != rows.len() as u64 {
        return Err("occupancy pyramid does not account for every CSV row".into());
    }
    println!(
        "converted {csv_rows} CSV rows into {} unique objects at {}",
        rows.len(),
        output.display()
    );
    Ok(())
}

fn merge_duplicate_ids(mut rows: Vec<CsvRow>) -> Result<Vec<CsvRow>> {
    rows.sort_unstable_by_key(|row| row.label_id);
    let mut unique: Vec<CsvRow> = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(previous) = unique.last_mut() else {
            unique.push(row);
            continue;
        };
        if previous.label_id != row.label_id {
            unique.push(row);
            continue;
        }
        let left_count = u64::from(previous.area_pixels);
        let right_count = u64::from(row.area_pixels);
        let total = left_count
            .checked_add(right_count)
            .ok_or("combined object area overflows u64")?;
        previous.area_pixels = u32::try_from(total)?;
        previous.centroid_y = (previous.centroid_y * left_count as f64
            + row.centroid_y * right_count as f64)
            / total as f64;
        previous.centroid_x = (previous.centroid_x * left_count as f64
            + row.centroid_x * right_count as f64)
            / total as f64;
        previous.dapi_mean = (previous.dapi_mean * left_count as f64
            + row.dapi_mean * right_count as f64)
            / total as f64;
        previous.area_um2 += row.area_um2;
        previous.dapi_min = previous.dapi_min.min(row.dapi_min);
        previous.dapi_max = previous.dapi_max.max(row.dapi_max);
    }
    Ok(unique)
}

fn tile_id(row: &CsvRow, grid_shape: &[u64]) -> usize {
    ((row.centroid_y as u64 / 1024) * grid_shape[1] + row.centroid_x as u64 / 1024) as usize
}

fn row_key(row: &CsvRow, grid_shape: &[u64]) -> (usize, u64, u64, u64) {
    (
        tile_id(row, grid_shape),
        float_key(row.centroid_y),
        float_key(row.centroid_x),
        row.label_id,
    )
}

fn float_key(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}
