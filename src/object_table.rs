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

/// What [`finalize_object_table`] buffers before it spills a sorted run.
pub const FINALIZE_BUFFER_BYTES: u64 = 64 << 20;

/// Project and finalise one fragment phase into a native object table.
///
/// The projected rows go through an [`ObjectTableBuilder`] holding at most
/// [`FINALIZE_BUFFER_BYTES`] of them: a table that fits is sorted and written
/// from memory, a larger one spilled in sorted runs and merged. Either way
/// the files are the same. `temporary_parent` should be on a filesystem with
/// space for the projected rows. Temporary files are removed on both success
/// and ordinary errors.
pub fn finalize_object_table<F>(
    env: &dyn Environment,
    stream: &str,
    phase: usize,
    volume: [usize; 3],
    input_schema: Schema,
    output_root: impl AsRef<Path>,
    temporary_parent: impl AsRef<Path>,
    spec: TableSpec,
    mut project: F,
) -> Result<TableReader>
where
    F: FnMut(&Row<'_>) -> Result<Option<Vec<ObjectValue>>>,
{
    let mut builder =
        ObjectTableBuilder::new(spec, Some(temporary_parent.as_ref()), FINALIZE_BUFFER_BYTES)?;
    fold_fragments(env, stream, &mut |key, bytes| {
        if key.phase != phase {
            return Ok(());
        }
        let mut table = Table::new(volume, input_schema.clone())?;
        table.write(key.block, bytes)?;
        table.seal()?;
        for row in table.scan(&Region::whole(&volume))? {
            if let Some(values) = project(&row)? {
                builder.push(&values)?;
            }
        }
        Ok(())
    })?;
    builder.finish(output_root)
}

/// Write rows held in memory as a native object table.
///
/// The bounded-memory path is [`finalize_object_table`], which folds fragment
/// streams, or an [`ObjectTableBuilder`] with a buffer limit, which takes rows
/// one at a time. This is for rows that already sit in memory because they were
/// assembled from somewhere else — for example one field of a time-lapse,
/// gathered from per-frame artifacts, with a few million rows at most. The
/// layout written is identical: rows in tile order, then by coordinates, then
/// by identity; the identity index sorted by id; the spatial index and its
/// occupancy pyramid. `spec.row_count` is set from `rows`.
pub fn write_object_rows(
    output_root: impl AsRef<Path>,
    spec: TableSpec,
    rows: Vec<Vec<ObjectValue>>,
) -> Result<TableReader> {
    let mut builder = ObjectTableBuilder::new(spec, None::<PathBuf>, u64::MAX)?;
    builder.reserve(rows.len());
    for values in rows {
        builder.push(&values)?;
    }
    builder.finish(output_root)
}

/// Rows pushed one at a time and written as a native object table, in bounded
/// memory.
///
/// A row is held as one word per column and one for its tile, nothing per row
/// beside them. While the rows fit `buffer_bytes` ([`Self::row_bytes`] each)
/// they are sorted and written from memory, as [`write_object_rows`] does;
/// past it every full buffer is sorted and spilled as a run under the
/// temporary directory, and [`Self::finish`] merges the runs as
/// [`finalize_object_table`] does. The rows are totally ordered (tile,
/// coordinates, identity, and identities are unique), so the table written is
/// the same file for file whatever the buffer and whatever the push order.
pub struct ObjectTableBuilder {
    spec: TableSpec,
    coordinate_columns: Vec<usize>,
    identity_column: usize,
    tile_counts: Vec<u64>,
    /// Per buffered row: its tile, then its values' bits.
    words: Vec<u64>,
    stride: usize,
    buffer_rows: usize,
    temporary_parent: Option<PathBuf>,
    temporary: Option<TemporaryDirectory>,
    runs: Vec<PathBuf>,
    total_rows: u64,
}

impl ObjectTableBuilder {
    /// A table of `spec`'s columns whose buffered rows may take
    /// `buffer_bytes`; spilled runs go under `temporary_parent`, which only a
    /// builder that can spill needs (`u64::MAX` bytes never spills).
    pub fn new(
        spec: TableSpec,
        temporary_parent: Option<impl AsRef<Path>>,
        buffer_bytes: u64,
    ) -> Result<Self> {
        spec.validate().map_err(native_error)?;
        let coordinate_columns = spec
            .spatial_index
            .coordinates
            .iter()
            .map(|coordinate| column_index(&spec, &coordinate.column))
            .collect::<Result<Vec<_>>>()?;
        let identity_column = column_index(&spec, &spec.identity)?;
        let tile_count = spec.tile_count().map_err(native_error)? as usize;
        let row_bytes = Self::row_bytes(&spec);
        let buffer_rows = usize::try_from(buffer_bytes / row_bytes)
            .unwrap_or(usize::MAX)
            .clamp(1, u32::MAX as usize);
        let temporary_parent = temporary_parent.map(|parent| parent.as_ref().to_path_buf());
        if buffer_bytes != u64::MAX && temporary_parent.is_none() {
            return Err(Error::invalid(
                "an object-table builder with a buffer limit spills runs to disk and needs a \
                 temporary directory",
            ));
        }
        Ok(Self {
            stride: spec.columns.len() + 1,
            spec,
            coordinate_columns,
            identity_column,
            tile_counts: vec![0u64; tile_count],
            words: Vec::new(),
            buffer_rows,
            temporary_parent,
            temporary: None,
            runs: Vec::new(),
            total_rows: 0,
        })
    }

    /// What one buffered row costs: a word per column and one for its tile,
    /// plus the sort order and the identity index built from it.
    pub fn row_bytes(spec: &TableSpec) -> u64 {
        Self::row_bytes_of(spec.columns.len())
    }

    /// [`Self::row_bytes`] of a table of `columns` columns.
    pub fn row_bytes_of(columns: usize) -> u64 {
        8 * (columns as u64 + 1) + 4 + 16
    }

    /// Room for `rows` more rows, as far as the buffer limit allows.
    pub fn reserve(&mut self, rows: usize) {
        let room = rows.min(self.buffer_rows.saturating_sub(self.buffered()));
        self.words.reserve_exact(room.saturating_mul(self.stride));
    }

    /// Rows pushed so far.
    pub fn rows(&self) -> u64 {
        self.total_rows
    }

    fn buffered(&self) -> usize {
        self.words.len() / self.stride
    }

    /// One row, a value per declared column in declaration order.
    pub fn push(&mut self, values: &[ObjectValue]) -> Result<()> {
        validate_values(&self.spec, values)?;
        let tile = tile_id(&self.spec, &self.coordinate_columns, values)?;
        for &column in &self.coordinate_columns {
            values[column].order_key()?;
        }
        values[self.identity_column].identity()?;
        self.tile_counts[tile as usize] = self.tile_counts[tile as usize]
            .checked_add(1)
            .ok_or_else(|| Error::invalid("object count overflow"))?;
        self.total_rows = self
            .total_rows
            .checked_add(1)
            .ok_or_else(|| Error::invalid("object-table row count overflow"))?;
        if self.words.len() + self.stride > self.words.capacity() {
            // Grown by doubling, but never past the limit: a doubled buffer
            // could hold twice what it was given.
            let limit = self.buffer_rows.saturating_mul(self.stride);
            let room = limit.saturating_sub(self.words.len()).max(self.stride);
            let grow = self.words.capacity().max(1024 * self.stride).min(room);
            self.words.reserve_exact(grow);
        }
        self.words.push(tile);
        self.words.extend(values.iter().map(|value| value.bits()));
        if self.buffered() >= self.buffer_rows {
            self.spill()?;
        }
        Ok(())
    }

    /// The buffered rows in table order.
    fn sorted(&self) -> Vec<u32> {
        let mut keys = Vec::with_capacity(self.coordinate_columns.len());
        for &column in &self.coordinate_columns {
            keys.push((column + 1, self.spec.columns[column].dtype));
        }
        let identity = self.identity_column + 1;
        let row = |index: u32| &self.words[index as usize * self.stride..][..self.stride];
        let mut order: Vec<u32> = (0..self.buffered() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            let (a, b) = (row(a), row(b));
            a[0].cmp(&b[0])
                .then_with(|| {
                    for &(at, dtype) in &keys {
                        let ordering = order_bits(dtype, a[at]).cmp(&order_bits(dtype, b[at]));
                        if ordering.is_ne() {
                            return ordering;
                        }
                    }
                    Ordering::Equal
                })
                .then_with(|| a[identity].cmp(&b[identity]))
        });
        order
    }

    /// Sort the buffer and write it as one run of [`finalize_object_table`]'s
    /// form: the key words (tile, coordinates' order keys, identity), then the
    /// values.
    fn spill(&mut self) -> Result<()> {
        if self.words.is_empty() {
            return Ok(());
        }
        if self.temporary.is_none() {
            let parent = self.temporary_parent.as_deref().ok_or_else(|| {
                Error::invalid("an object-table builder without a temporary directory spilled")
            })?;
            self.temporary = Some(TemporaryDirectory::create(parent)?);
        }
        let order = self.sorted();
        let directory = &self.temporary.as_ref().expect("created above").path;
        let path = directory.join(format!("spatial-{}.run", self.runs.len()));
        let mut out = BufWriter::new(File::create(&path).map_err(Error::backend)?);
        for index in order {
            let row = &self.words[index as usize * self.stride..][..self.stride];
            let mut put = |word: u64| out.write_all(&word.to_le_bytes()).map_err(Error::backend);
            put(row[0])?;
            for &column in &self.coordinate_columns {
                put(order_bits(self.spec.columns[column].dtype, row[column + 1]))?;
            }
            put(row[self.identity_column + 1])?;
            for &word in &row[1..] {
                put(word)?;
            }
        }
        out.flush().map_err(Error::backend)?;
        self.runs.push(path);
        self.words.clear();
        Ok(())
    }

    /// Write the table at `output_root`.
    pub fn finish(mut self, output_root: impl AsRef<Path>) -> Result<TableReader> {
        self.spec.row_count = self.total_rows;
        if self.runs.is_empty() {
            return self.finish_in_memory(output_root.as_ref());
        }
        self.spill()?;
        let temporary = self
            .temporary
            .take()
            .expect("a spilled builder has a directory");
        let key_width = self.coordinate_columns.len() + 2;
        let runs = std::mem::take(&mut self.runs);
        let spatial_run = collapse_runs(
            &temporary.path,
            "spatial",
            runs,
            key_width,
            self.spec.columns.len(),
        )?;
        let writer =
            TableWriter::create(output_root.as_ref(), self.spec.clone()).map_err(native_error)?;
        let mut identity_runs = Vec::new();
        if let Some(path) = spatial_run {
            write_spatial_rows(
                &writer,
                &self.spec,
                &path,
                key_width,
                self.identity_column,
                &temporary.path,
                &mut identity_runs,
            )?;
        }
        let identity_run = collapse_runs(&temporary.path, "identity", identity_runs, 1, 1)?;
        if let Some(path) = identity_run {
            write_identity_rows(&writer, &self.spec, &path)?;
        }
        write_spatial_index(&writer, &self.tile_counts)?;
        writer.finish().map_err(native_error)
    }

    fn finish_in_memory(self, output_root: &Path) -> Result<TableReader> {
        let order = self.sorted();
        let spec = &self.spec;
        let writer = TableWriter::create(output_root, spec.clone()).map_err(native_error)?;
        let chunk = spec.row_chunk as usize;
        let row = |index: u32| &self.words[index as usize * self.stride..][..self.stride];
        let mut words = Vec::with_capacity(chunk.min(order.len()));
        for (index, block) in order.chunks(chunk).enumerate() {
            let start = (index * chunk) as u64;
            for (column_number, column) in spec.columns.iter().enumerate() {
                words.clear();
                words.extend(block.iter().map(|&at| row(at)[column_number + 1]));
                write_column(&writer, &column.name, column.dtype, start, &words)?;
            }
        }
        let identity = self.identity_column + 1;
        let mut identities: Vec<(u64, u64)> = order
            .iter()
            .enumerate()
            .map(|(position, &at)| (row(at)[identity], position as u64))
            .collect();
        drop(order);
        identities.sort_unstable();
        for pair in identities.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(Error::invalid(format!(
                    "object-table identity {} occurs more than once",
                    pair[0].0
                )));
            }
        }
        let mut ids = Vec::with_capacity(chunk.min(identities.len()));
        let mut positions = Vec::with_capacity(chunk.min(identities.len()));
        for (index, block) in identities.chunks(chunk).enumerate() {
            ids.clear();
            positions.clear();
            ids.extend(block.iter().map(|&(id, _)| id));
            positions.extend(block.iter().map(|&(_, position)| position));
            writer
                .write_identity_index((index * chunk) as u64, &ids, &positions)
                .map_err(native_error)?;
        }
        write_spatial_index(&writer, &self.tile_counts)?;
        writer.finish().map_err(native_error)
    }
}

/// The spatial index from the per-tile counts: each tile's first row and
/// count.
fn write_spatial_index(writer: &TableWriter, tile_counts: &[u64]) -> Result<()> {
    let mut starts = vec![0u64; tile_counts.len()];
    let mut next = 0u64;
    for (start, count) in starts.iter_mut().zip(tile_counts) {
        *start = next;
        next = next
            .checked_add(*count)
            .ok_or_else(|| Error::invalid("spatial index overflow"))?;
    }
    writer
        .write_spatial_index(&starts, tile_counts)
        .map_err(native_error)
}

/// [`ObjectValue::order_key`] of a value already stored as its bits (and
/// already checked to be finite).
fn order_bits(dtype: DType, bits: u64) -> u64 {
    match dtype {
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => bits,
        DType::I32 => u64::from((bits as u32) ^ (1 << 31)),
        DType::I64 => bits ^ (1 << 63),
        DType::F32 => {
            let bits = bits as u32;
            u64::from(if bits & (1 << 31) != 0 {
                !bits
            } else {
                bits ^ (1 << 31)
            })
        }
        DType::F64 => {
            if bits & (1 << 63) != 0 {
                !bits
            } else {
                bits ^ (1 << 63)
            }
        }
    }
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
            downsample: None,
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

    #[test]
    fn in_memory_rows_are_written_in_tile_coordinate_identity_order() {
        let coordinate = |axis: &str| CoordinateColumn {
            axis: axis.into(),
            column: axis.into(),
        };
        let columns = vec![
            ColumnSpec::new("id", DType::U64, ColumnRole::Identity),
            ColumnSpec::new("t", DType::F64, ColumnRole::Coordinate),
            ColumnSpec::new("y", DType::F64, ColumnRole::Coordinate),
            ColumnSpec::new("x", DType::F64, ColumnRole::Coordinate),
        ];
        let spatial_index = SpatialIndexSpec {
            coordinates: vec![coordinate("t"), coordinate("y"), coordinate("x")],
            tile_shape: vec![1, 4, 4],
            grid_shape: vec![2, 2, 2],
            tile_order: "row_major".into(),
            within_tile_order: "lexicographic_coordinates_then_identity".into(),
            downsample: Some(vec![false, true, true]),
        };
        let spec = TableSpec::new(0, 2, "../../", "id", columns, spatial_index);
        let row = |id, t, y, x| {
            vec![
                ObjectValue::U64(id),
                ObjectValue::F64(t),
                ObjectValue::F64(y),
                ObjectValue::F64(x),
            ]
        };
        let rows = vec![
            row(70, 1.0, 1.0, 1.0),
            row(5, 0.0, 6.0, 6.0),
            row(9, 0.0, 1.0, 2.0),
            row(8, 0.0, 1.0, 1.0),
        ];
        let scratch = TemporaryDirectory::create(&std::env::temp_dir()).unwrap();
        let reader = write_object_rows(scratch.path.join("table"), spec.clone(), rows).unwrap();
        assert_eq!(reader.read_u64("id", 0..4).unwrap(), [8, 9, 5, 70]);
        assert_eq!(
            reader.read_identity_index(0..4).unwrap(),
            (vec![5, 8, 9, 70], vec![2, 0, 1, 3])
        );
        assert_eq!(reader.occupancy_shapes(), [vec![2, 2, 2], vec![2, 1, 1]]);
        assert_eq!(
            reader.read_occupancy(1, &[0, 0, 0], &[2, 1, 1]).unwrap(),
            [3, 1]
        );
        let duplicate = vec![row(1, 0.0, 0.0, 0.0), row(1, 1.0, 0.0, 0.0)];
        assert!(write_object_rows(scratch.path.join("dup"), spec, duplicate).is_err());
    }

    /// Every file under `root`, by its path below it, with its bytes.
    fn files(root: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
        let mut out = std::collections::BTreeMap::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(directory) = stack.pop() {
            for entry in fs::read_dir(&directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let relative = path.strip_prefix(root).unwrap().to_path_buf();
                    out.insert(relative, fs::read(&path).unwrap());
                }
            }
        }
        out
    }

    #[test]
    fn a_spilling_builder_writes_the_table_rows_in_memory_write() {
        let coordinate = |axis: &str| CoordinateColumn {
            axis: axis.into(),
            column: axis.into(),
        };
        let columns = vec![
            ColumnSpec::new("id", DType::U64, ColumnRole::Identity),
            ColumnSpec::new("z", DType::F64, ColumnRole::Coordinate),
            ColumnSpec::new("y", DType::F32, ColumnRole::Coordinate),
            ColumnSpec::new("x", DType::I32, ColumnRole::Coordinate),
            ColumnSpec::new("flag", DType::U8, ColumnRole::Category),
            ColumnSpec::new("value", DType::F64, ColumnRole::Measurement),
            ColumnSpec::new("label", DType::I64, ColumnRole::Category),
        ];
        let spatial_index = SpatialIndexSpec {
            coordinates: vec![coordinate("z"), coordinate("y"), coordinate("x")],
            tile_shape: vec![4, 4, 4],
            grid_shape: vec![3, 3, 3],
            tile_order: "row_major".into(),
            within_tile_order: "lexicographic_coordinates_then_identity".into(),
            downsample: None,
        };
        let spec = TableSpec::new(0, 7, "../../", "id", columns, spatial_index);
        // Pseudo-random rows with repeated positions, so the identity breaks
        // ties, pushed in an order unrelated to the table's.
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let rows: Vec<Vec<ObjectValue>> = (0..300u64)
            .map(|id| {
                vec![
                    ObjectValue::U64((id * 7919) % 1000),
                    ObjectValue::F64((next() % 12) as f64 * 0.75),
                    ObjectValue::F32((next() % 12) as f32),
                    ObjectValue::I32((next() % 12) as i32),
                    ObjectValue::U8((next() % 2) as u8),
                    ObjectValue::F64(next() as f64 / 3.0),
                    ObjectValue::I64(next() as i64 - 1000),
                ]
            })
            .collect();
        let scratch = TemporaryDirectory::create(&std::env::temp_dir()).unwrap();
        let in_memory = scratch.path.join("in-memory");
        write_object_rows(&in_memory, spec.clone(), rows.clone()).unwrap();
        let row_bytes = ObjectTableBuilder::row_bytes(&spec);
        for buffered in [1u64, 2, 13, 64, 299, 300, 1000] {
            let spilled = scratch.path.join(format!("spilled-{buffered}"));
            let mut builder =
                ObjectTableBuilder::new(spec.clone(), Some(&scratch.path), buffered * row_bytes)
                    .unwrap();
            for row in rows.iter().rev() {
                builder.push(row).unwrap();
            }
            assert_eq!(builder.rows(), 300);
            builder.finish(&spilled).unwrap();
            assert_eq!(
                files(&in_memory),
                files(&spilled),
                "{buffered} rows buffered"
            );
        }
        // The temporary runs are gone, and a duplicate identity is refused
        // whether or not the builder spilled.
        assert_eq!(
            fs::read_dir(&scratch.path)
                .unwrap()
                .filter(|entry| entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".blockflow-object-table"))
                .count(),
            0
        );
        for buffered in [2u64, u64::MAX / row_bytes] {
            let mut builder = ObjectTableBuilder::new(
                spec.clone(),
                Some(&scratch.path),
                buffered.saturating_mul(row_bytes),
            )
            .unwrap();
            for row in rows.iter().chain(&rows[..1]) {
                builder.push(row).unwrap();
            }
            assert!(builder.finish(scratch.path.join("duplicate")).is_err());
        }
        assert!(ObjectTableBuilder::new(spec, None::<PathBuf>, 1 << 20).is_err());
    }
}
