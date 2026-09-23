//! Finalise Blockflow row fragments into a spatially indexed object table.
//!
//! Planner blocks are an execution detail. This module projects every fragment
//! row into the native table schema, partitions it by the table's fixed spatial
//! grid, and externally sorts bounded runs. The final merge writes each Zarr row
//! chunk once. Identity indexing uses the same bounded merge, and occupancy
//! levels are derived by `ngff-object-table` from the spatial counts.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use ngff_object_table::TableWriter;
pub use ngff_object_table::{
    ColumnRole, ColumnSpec, CoordinateColumn, DType, SpatialIndexSpec, TableReader, TableSpec,
};

use crate::env::Environment;
use crate::fragment::fold_fragments;
use crate::region::Region;
use crate::table::{Row, Schema, Table};
use crate::{Error, Result};

const MERGE_FAN_IN: usize = 64;

/// A projected value for one native object-table column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObjectValue {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl ObjectValue {
    fn dtype(self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    fn bits(self) -> u64 {
        match self {
            Self::U8(v) => u64::from(v),
            Self::U16(v) => u64::from(v),
            Self::U32(v) => u64::from(v),
            Self::U64(v) => v,
            Self::I32(v) => u64::from(v as u32),
            Self::I64(v) => v as u64,
            Self::F32(v) => u64::from(v.to_bits()),
            Self::F64(v) => v.to_bits(),
        }
    }

    fn order_key(self) -> Result<u64> {
        Ok(match self {
            Self::U8(v) => u64::from(v),
            Self::U16(v) => u64::from(v),
            Self::U32(v) => u64::from(v),
            Self::U64(v) => v,
            Self::I32(v) => u64::from((v as u32) ^ (1 << 31)),
            Self::I64(v) => (v as u64) ^ (1 << 63),
            Self::F32(v) => {
                if !v.is_finite() {
                    return Err(Error::invalid("object-table coordinates must be finite"));
                }
                let bits = v.to_bits();
                u64::from(if bits & (1 << 31) != 0 {
                    !bits
                } else {
                    bits ^ (1 << 31)
                })
            }
            Self::F64(v) => {
                if !v.is_finite() {
                    return Err(Error::invalid("object-table coordinates must be finite"));
                }
                let bits = v.to_bits();
                if bits & (1 << 63) != 0 {
                    !bits
                } else {
                    bits ^ (1 << 63)
                }
            }
        })
    }

    fn tile_index(self, tile_shape: u64) -> Result<u64> {
        let invalid = || Error::invalid("object-table coordinates must be finite and nonnegative");
        Ok(match self {
            Self::U8(v) => u64::from(v) / tile_shape,
            Self::U16(v) => u64::from(v) / tile_shape,
            Self::U32(v) => u64::from(v) / tile_shape,
            Self::U64(v) => v / tile_shape,
            Self::I32(v) => u64::try_from(v).map_err(|_| invalid())? / tile_shape,
            Self::I64(v) => u64::try_from(v).map_err(|_| invalid())? / tile_shape,
            Self::F32(v) => {
                if !v.is_finite() || v < 0.0 {
                    return Err(invalid());
                }
                (f64::from(v) / tile_shape as f64).floor() as u64
            }
            Self::F64(v) => {
                if !v.is_finite() || v < 0.0 {
                    return Err(invalid());
                }
                (v / tile_shape as f64).floor() as u64
            }
        })
    }

    fn identity(self) -> Result<u64> {
        match self {
            Self::U64(value) => Ok(value),
            _ => Err(Error::invalid("object-table identity column must be u64")),
        }
    }
}

#[derive(Clone)]
struct Record {
    key: Vec<u64>,
    values: Vec<u64>,
}

/// Project and finalise one fragment phase into a native object table.
///
/// `temporary_parent` should be on a filesystem with space for the projected
/// rows. Temporary files are removed on both success and ordinary errors.
pub fn finalize_object_table<F>(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
    input_schema: Schema,
    output_root: impl AsRef<Path>,
    temporary_parent: impl AsRef<Path>,
    mut spec: TableSpec,
    mut project: F,
) -> Result<TableReader>
where
    F: FnMut(&Row<'_>) -> Result<Option<Vec<ObjectValue>>>,
{
    spec.validate().map_err(native_error)?;
    let coordinate_columns = spec
        .spatial_index
        .coordinates
        .iter()
        .map(|coordinate| column_index(&spec, &coordinate.column))
        .collect::<Result<Vec<_>>>()?;
    let identity_column = column_index(&spec, &spec.identity)?;
    let tile_count = spec.tile_count().map_err(native_error)? as usize;
    let mut tile_counts = vec![0u64; tile_count];
    let temporary = TemporaryDirectory::create(temporary_parent.as_ref())?;
    let mut runs = Vec::new();
    let mut total_rows = 0u64;

    fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        let mut table = Table::new(volume, input_schema.clone())?;
        table.write(key.block, bytes)?;
        table.seal()?;
        let mut records = Vec::new();
        for row in table.scan(&Region::whole(&volume))? {
            let Some(values) = project(&row)? else {
                continue;
            };
            validate_values(&spec, &values)?;
            let tile = tile_id(&spec, &coordinate_columns, &values)?;
            tile_counts[tile as usize] = tile_counts[tile as usize]
                .checked_add(1)
                .ok_or_else(|| Error::invalid("object count overflow"))?;
            let mut sort_key = Vec::with_capacity(coordinate_columns.len() + 2);
            sort_key.push(tile);
            for &column in &coordinate_columns {
                sort_key.push(values[column].order_key()?);
            }
            sort_key.push(values[identity_column].identity()?);
            records.push(Record {
                key: sort_key,
                values: values.into_iter().map(ObjectValue::bits).collect(),
            });
        }
        if !records.is_empty() {
            records.sort_unstable_by(|a, b| a.key.cmp(&b.key));
            let path = temporary.path.join(format!("spatial-{}.run", runs.len()));
            write_run(&path, &records)?;
            total_rows = total_rows
                .checked_add(records.len() as u64)
                .ok_or_else(|| Error::invalid("object-table row count overflow"))?;
            runs.push(path);
        }
        Ok(())
    })?;

    let key_width = coordinate_columns.len() + 2;
    let value_width = spec.columns.len();
    let spatial_run = collapse_runs(&temporary.path, "spatial", runs, key_width, value_width)?;
    spec.row_count = total_rows;
    let writer = TableWriter::create(output_root.as_ref(), spec.clone()).map_err(native_error)?;
    let mut identity_runs = Vec::new();
    if let Some(path) = spatial_run {
        write_spatial_rows(
            &writer,
            &spec,
            &path,
            key_width,
            identity_column,
            &temporary.path,
            &mut identity_runs,
        )?;
    }
    let identity_run = collapse_runs(&temporary.path, "identity", identity_runs, 1, 1)?;
    if let Some(path) = identity_run {
        write_identity_rows(&writer, &spec, &path)?;
    }
    let mut starts = vec![0u64; tile_counts.len()];
    let mut next = 0u64;
    for (start, count) in starts.iter_mut().zip(&tile_counts) {
        *start = next;
        next = next
            .checked_add(*count)
            .ok_or_else(|| Error::invalid("spatial index overflow"))?;
    }
    writer
        .write_spatial_index(&starts, &tile_counts)
        .map_err(native_error)?;
    writer.finish().map_err(native_error)
}

fn column_index(spec: &TableSpec, name: &str) -> Result<usize> {
    spec.columns
        .iter()
        .position(|column| column.name == name)
        .ok_or_else(|| Error::invalid(format!("object-table column {name:?} is not declared")))
}

fn validate_values(spec: &TableSpec, values: &[ObjectValue]) -> Result<()> {
    if values.len() != spec.columns.len() {
        return Err(Error::invalid(format!(
            "object-table projection returned {} columns; schema declares {}",
            values.len(),
            spec.columns.len()
        )));
    }
    for (column, value) in spec.columns.iter().zip(values) {
        if column.dtype != value.dtype() {
            return Err(Error::invalid(format!(
                "object-table column {:?} is {:?}, but projection returned {:?}",
                column.name,
                column.dtype,
                value.dtype()
            )));
        }
    }
    Ok(())
}

fn tile_id(spec: &TableSpec, columns: &[usize], values: &[ObjectValue]) -> Result<u64> {
    let mut linear = 0u64;
    for (axis, &column) in columns.iter().enumerate() {
        let tile = values[column].tile_index(spec.spatial_index.tile_shape[axis])?;
        let grid = spec.spatial_index.grid_shape[axis];
        if tile >= grid {
            return Err(Error::invalid(format!(
                "a coordinate maps to tile {tile} outside axis {axis} grid length {grid}"
            )));
        }
        linear = linear
            .checked_mul(grid)
            .and_then(|value| value.checked_add(tile))
            .ok_or_else(|| Error::invalid("spatial tile index overflow"))?;
    }
    Ok(linear)
}

fn write_run(path: &Path, records: &[Record]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path).map_err(Error::backend)?);
    for record in records {
        for word in record.key.iter().chain(&record.values) {
            out.write_all(&word.to_le_bytes()).map_err(Error::backend)?;
        }
    }
    out.flush().map_err(Error::backend)
}

struct RunReader {
    input: BufReader<File>,
    key_width: usize,
    value_width: usize,
}

impl RunReader {
    fn open(path: &Path, key_width: usize, value_width: usize) -> Result<Self> {
        Ok(Self {
            input: BufReader::new(File::open(path).map_err(Error::backend)?),
            key_width,
            value_width,
        })
    }

    fn next(&mut self, reuse: Option<Record>) -> Result<Option<Record>> {
        let mut record = reuse.unwrap_or_else(|| Record {
            key: vec![0; self.key_width],
            values: vec![0; self.value_width],
        });
        record.key.resize(self.key_width, 0);
        record.values.resize(self.value_width, 0);
        let Some(first) = read_first_word(&mut self.input)? else {
            return Ok(None);
        };
        record.key[0] = first;
        for word in record.key[1..].iter_mut().chain(&mut record.values) {
            let mut bytes = [0u8; 8];
            self.input.read_exact(&mut bytes).map_err(|error| {
                if error.kind() == std::io::ErrorKind::UnexpectedEof {
                    Error::invalid("truncated object-table temporary run")
                } else {
                    Error::backend(error)
                }
            })?;
            *word = u64::from_le_bytes(bytes);
        }
        Ok(Some(record))
    }
}

fn read_first_word(input: &mut BufReader<File>) -> Result<Option<u64>> {
    let mut bytes = [0u8; 8];
    if input.read(&mut bytes[..1]).map_err(Error::backend)? == 0 {
        return Ok(None);
    }
    input.read_exact(&mut bytes[1..]).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::invalid("truncated object-table temporary run")
        } else {
            Error::backend(error)
        }
    })?;
    Ok(Some(u64::from_le_bytes(bytes)))
}

struct HeapRecord {
    record: Record,
    reader: usize,
}

impl PartialEq for HeapRecord {
    fn eq(&self, other: &Self) -> bool {
        self.record.key == other.record.key && self.reader == other.reader
    }
}
impl Eq for HeapRecord {}
impl PartialOrd for HeapRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .record
            .key
            .cmp(&self.record.key)
            .then_with(|| other.reader.cmp(&self.reader))
    }
}

fn merge_runs(
    paths: &[PathBuf],
    output: &Path,
    key_width: usize,
    value_width: usize,
) -> Result<()> {
    let mut readers = paths
        .iter()
        .map(|path| RunReader::open(path, key_width, value_width))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = BinaryHeap::new();
    for (reader, input) in readers.iter_mut().enumerate() {
        if let Some(record) = input.next(None)? {
            heap.push(HeapRecord { record, reader });
        }
    }
    let mut out = BufWriter::new(File::create(output).map_err(Error::backend)?);
    while let Some(item) = heap.pop() {
        for word in item.record.key.iter().chain(&item.record.values) {
            out.write_all(&word.to_le_bytes()).map_err(Error::backend)?;
        }
        if let Some(record) = readers[item.reader].next(Some(item.record))? {
            heap.push(HeapRecord {
                record,
                reader: item.reader,
            });
        }
    }
    out.flush().map_err(Error::backend)
}

fn collapse_runs(
    directory: &Path,
    prefix: &str,
    mut runs: Vec<PathBuf>,
    key_width: usize,
    value_width: usize,
) -> Result<Option<PathBuf>> {
    let mut generation = 0usize;
    while runs.len() > 1 {
        let mut next = Vec::new();
        for (group, inputs) in runs.chunks(MERGE_FAN_IN).enumerate() {
            let output = directory.join(format!("{prefix}-merge-{generation}-{group}.run"));
            merge_runs(inputs, &output, key_width, value_width)?;
            for input in inputs {
                fs::remove_file(input).map_err(Error::backend)?;
            }
            next.push(output);
        }
        runs = next;
        generation += 1;
    }
    Ok(runs.pop())
}

fn write_spatial_rows(
    writer: &TableWriter,
    spec: &TableSpec,
    path: &Path,
    key_width: usize,
    identity_column: usize,
    temporary: &Path,
    identity_runs: &mut Vec<PathBuf>,
) -> Result<()> {
    let chunk = spec.row_chunk as usize;
    let mut input = RunReader::open(path, key_width, spec.columns.len())?;
    let mut buffers = vec![Vec::with_capacity(chunk); spec.columns.len()];
    let mut row_start = 0u64;
    let mut reuse = None;
    loop {
        let mut count = 0usize;
        while count < chunk {
            let Some(record) = input.next(reuse.take())? else {
                break;
            };
            for (buffer, &value) in buffers.iter_mut().zip(&record.values) {
                buffer.push(value);
            }
            reuse = Some(record);
            count += 1;
        }
        if count == 0 {
            break;
        }
        for (column, values) in spec.columns.iter().zip(&buffers) {
            write_column(
                writer,
                column.name.as_str(),
                column.dtype,
                row_start,
                values,
            )?;
        }
        let mut identities = buffers[identity_column]
            .iter()
            .enumerate()
            .map(|(offset, id)| Record {
                key: vec![*id],
                values: vec![row_start + offset as u64],
            })
            .collect::<Vec<_>>();
        identities.sort_unstable_by(|a, b| a.key.cmp(&b.key));
        let identity_path = temporary.join(format!("identity-{}.run", identity_runs.len()));
        write_run(&identity_path, &identities)?;
        identity_runs.push(identity_path);
        row_start += count as u64;
        for buffer in &mut buffers {
            buffer.clear();
        }
    }
    Ok(())
}

fn write_identity_rows(writer: &TableWriter, spec: &TableSpec, path: &Path) -> Result<()> {
    let mut input = RunReader::open(path, 1, 1)?;
    let mut start = 0u64;
    let mut previous = None;
    let mut reuse = None;
    loop {
        let mut ids = Vec::with_capacity(spec.row_chunk as usize);
        let mut rows = Vec::with_capacity(spec.row_chunk as usize);
        while ids.len() < spec.row_chunk as usize {
            let Some(record) = input.next(reuse.take())? else {
                break;
            };
            let id = record.key[0];
            if previous == Some(id) {
                return Err(Error::invalid(format!(
                    "object-table identity {id} occurs more than once"
                )));
            }
            previous = Some(id);
            ids.push(id);
            rows.push(record.values[0]);
            reuse = Some(record);
        }
        if ids.is_empty() {
            break;
        }
        writer
            .write_identity_index(start, &ids, &rows)
            .map_err(native_error)?;
        start += ids.len() as u64;
    }
    Ok(())
}

fn write_column(
    writer: &TableWriter,
    name: &str,
    dtype: DType,
    start: u64,
    words: &[u64],
) -> Result<()> {
    macro_rules! write_as {
        ($method:ident, $convert:expr) => {{
            let values = words.iter().copied().map($convert).collect::<Vec<_>>();
            writer.$method(name, start, &values).map_err(native_error)
        }};
    }
    match dtype {
        DType::U8 => write_as!(write_u8, |v| v as u8),
        DType::U16 => write_as!(write_u16, |v| v as u16),
        DType::U32 => write_as!(write_u32, |v| v as u32),
        DType::U64 => writer.write_u64(name, start, words).map_err(native_error),
        DType::I32 => write_as!(write_i32, |v| v as u32 as i32),
        DType::I64 => write_as!(write_i64, |v| v as i64),
        DType::F32 => write_as!(write_f32, |v| f32::from_bits(v as u32)),
        DType::F64 => write_as!(write_f64, f64::from_bits),
    }
}

fn native_error(error: ngff_object_table::Error) -> Error {
    Error::backend(error)
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn create(parent: &Path) -> Result<Self> {
        fs::create_dir_all(parent).map_err(Error::backend)?;
        for suffix in 0..1000u32 {
            let path = parent.join(format!(
                ".blockflow-object-table-{}-{suffix}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(Error::backend(error)),
            }
        }
        Err(Error::invalid(
            "could not allocate a temporary finalizer directory",
        ))
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::Array3;

    use super::*;
    use crate::env::ArrayEnvironment;
    use crate::sidecar::Lifecycle;
    use crate::table::{Column, RowBuilder, Value};

    #[test]
    fn fragments_are_partitioned_independently_of_their_producer_blocks() {
        let schema = Schema::new(vec![Column::u64("id"), Column::u64("area")]).unwrap();
        let mut first = RowBuilder::new(Arc::new(schema.clone()));
        first
            .push([0, 6, 1], &[Value::U64(30), Value::U64(3)])
            .unwrap();
        first
            .push([0, 1, 6], &[Value::U64(10), Value::U64(1)])
            .unwrap();
        let mut second = RowBuilder::new(Arc::new(schema.clone()));
        second
            .push([0, 2, 2], &[Value::U64(40), Value::U64(4)])
            .unwrap();
        second
            .push([0, 1, 1], &[Value::U64(20), Value::U64(2)])
            .unwrap();

        let env =
            ArrayEnvironment::new(Array3::<f32>::zeros((1, 8, 8)).into(), 1, [1, 4, 4]).unwrap();
        env.declare_sidecar("objects", Lifecycle::Persistent)
            .unwrap();
        env.write_sidecar("objects", 0, [0, 0, 0], &first.encode())
            .unwrap();
        env.write_sidecar("objects", 0, [0, 1, 1], &second.encode())
            .unwrap();

        let columns = vec![
            ColumnSpec::new("id", DType::U64, ColumnRole::Identity),
            ColumnSpec::new("y", DType::F32, ColumnRole::Coordinate),
            ColumnSpec::new("x", DType::F32, ColumnRole::Coordinate),
            ColumnSpec::new("area", DType::U32, ColumnRole::Measurement),
        ];
        let spatial_index = SpatialIndexSpec {
            coordinates: vec![
                CoordinateColumn {
                    axis: "y".into(),
                    column: "y".into(),
                },
                CoordinateColumn {
                    axis: "x".into(),
                    column: "x".into(),
                },
            ],
            tile_shape: vec![4, 4],
            grid_shape: vec![2, 2],
            tile_order: "row_major".into(),
            within_tile_order: "lexicographic_coordinates_then_identity".into(),
        };
        let spec = TableSpec::new(0, 2, "../../", "id", columns, spatial_index);
        let scratch = TemporaryDirectory::create(&std::env::temp_dir()).unwrap();
        let output = scratch.path.join("table");
        let reader = finalize_object_table(
            &env,
            "objects",
            0,
            [1, 8, 8],
            schema,
            &output,
            &scratch.path,
            spec,
            |row| {
                let at = row.at();
                Ok(Some(vec![
                    ObjectValue::U64(row.u64(0)?),
                    ObjectValue::F32(at[1] as f32),
                    ObjectValue::F32(at[2] as f32),
                    ObjectValue::U32(row.u64(1)? as u32),
                ]))
            },
        )
        .unwrap();

        assert_eq!(reader.read_u64("id", 0..4).unwrap(), [20, 40, 10, 30]);
        assert_eq!(
            reader.read_identity_index(0..4).unwrap(),
            (vec![10, 20, 30, 40], vec![2, 0, 3, 1])
        );
        assert_eq!(
            reader.read_spatial_index().unwrap(),
            (vec![0, 2, 3, 4], vec![2, 1, 1, 0])
        );
        assert_eq!(reader.read_occupancy(1, &[0, 0], &[1, 1]).unwrap(), [4]);
    }
}
