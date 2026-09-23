//! Typed, spatially indexed object tables in Zarr v3.
//!
//! The storage grid is part of the table and is independent of the compute grid
//! used to produce it. Writers fill complete row chunks and publish the root
//! metadata last. Readers can then load selected columns and spatial tile ranges
//! without materialising the whole table.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value as Json};
use zarrs::array::codec::api::BytesToBytesCodecTraits;
use zarrs::array::codec::GzipCodec;
use zarrs::array::data_type;
use zarrs::array::{
    Array as ZarrArray, ArrayBuilder, ArraySubset, DataType, Element, ElementOwned, FillValue,
};
use zarrs::filesystem::FilesystemStore;

const METADATA_KEY: &str = "object_table";
pub const FORMAT_VERSION: u32 = 1;

type Store = FilesystemStore;
type Array = ZarrArray<Store>;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    Storage(String),
}

impl Error {
    fn storage(error: impl fmt::Display) -> Self {
        Self::Storage(error.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) | Self::Storage(message) => out.write_str(message),
            Self::Io(error) => error.fmt(out),
            Self::Json(error) => error.fmt(out),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    U8,
    U16,
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
}

impl DType {
    fn zarr(self) -> DataType {
        match self {
            Self::U8 => data_type::uint8(),
            Self::U16 => data_type::uint16(),
            Self::U32 => data_type::uint32(),
            Self::U64 => data_type::uint64(),
            Self::I32 => data_type::int32(),
            Self::I64 => data_type::int64(),
            Self::F32 => data_type::float32(),
            Self::F64 => data_type::float64(),
        }
    }

    fn fill(self) -> FillValue {
        match self {
            Self::U8 => FillValue::from(0u8),
            Self::U16 => FillValue::from(0u16),
            Self::U32 => FillValue::from(0u32),
            Self::U64 => FillValue::from(0u64),
            Self::I32 => FillValue::from(0i32),
            Self::I64 => FillValue::from(0i64),
            Self::F32 => FillValue::from(0f32),
            Self::F64 => FillValue::from(0f64),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnRole {
    Identity,
    Coordinate,
    Measurement,
    Intensity,
    Category,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    pub dtype: DType,
    pub role: ColumnRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_range: Option<[f64; 2]>,
    #[serde(default)]
    pub nullable: bool,
}

impl ColumnSpec {
    pub fn new(name: impl Into<String>, dtype: DType, role: ColumnRole) -> Self {
        Self {
            name: name.into(),
            dtype,
            role,
            display_name: None,
            unit: None,
            valid_range: None,
            nullable: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinateColumn {
    pub axis: String,
    pub column: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialIndexSpec {
    pub coordinates: Vec<CoordinateColumn>,
    pub tile_shape: Vec<u64>,
    pub grid_shape: Vec<u64>,
    #[serde(default = "default_tile_order")]
    pub tile_order: String,
    #[serde(default = "default_row_order")]
    pub within_tile_order: String,
}

fn default_tile_order() -> String {
    "row_major".to_owned()
}

fn default_row_order() -> String {
    "lexicographic_coordinates_then_identity".to_owned()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSpec {
    pub version: u32,
    pub row_count: u64,
    pub row_chunk: u64,
    pub coordinate_space: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    pub identity: String,
    pub columns: Vec<ColumnSpec>,
    pub spatial_index: SpatialIndexSpec,
}

impl TableSpec {
    pub fn new(
        row_count: u64,
        row_chunk: u64,
        coordinate_space: impl Into<String>,
        identity: impl Into<String>,
        columns: Vec<ColumnSpec>,
        spatial_index: SpatialIndexSpec,
    ) -> Self {
        Self {
            version: FORMAT_VERSION,
            row_count,
            row_chunk,
            coordinate_space: coordinate_space.into(),
            region: None,
            identity: identity.into(),
            columns,
            spatial_index,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != FORMAT_VERSION {
            return Err(Error::Invalid(format!(
                "object-table version {} is unsupported; expected {FORMAT_VERSION}",
                self.version
            )));
        }
        if self.row_chunk == 0 {
            return Err(Error::Invalid("row_chunk must be positive".into()));
        }
        if self.coordinate_space.is_empty() {
            return Err(Error::Invalid("coordinate_space must not be empty".into()));
        }
        let mut names = BTreeSet::new();
        for column in &self.columns {
            validate_component(&column.name, "column")?;
            if !names.insert(column.name.as_str()) {
                return Err(Error::Invalid(format!(
                    "column {:?} is declared more than once",
                    column.name
                )));
            }
            if let Some([lo, hi]) = column.valid_range {
                if !lo.is_finite() || !hi.is_finite() || lo > hi {
                    return Err(Error::Invalid(format!(
                        "column {:?} has invalid range [{lo}, {hi}]",
                        column.name
                    )));
                }
            }
        }
        let identity = self
            .columns
            .iter()
            .find(|column| column.name == self.identity)
            .ok_or_else(|| {
                Error::Invalid(format!("identity column {:?} is absent", self.identity))
            })?;
        if identity.dtype != DType::U64 || identity.role != ColumnRole::Identity {
            return Err(Error::Invalid(
                "the identity column must have identity role and uint64 type".into(),
            ));
        }
        let spatial = &self.spatial_index;
        if spatial.coordinates.is_empty()
            || spatial.coordinates.len() != spatial.tile_shape.len()
            || spatial.coordinates.len() != spatial.grid_shape.len()
        {
            return Err(Error::Invalid(
                "spatial coordinates, tile_shape, and grid_shape must have the same nonzero rank"
                    .into(),
            ));
        }
        if spatial.tile_shape.contains(&0) || spatial.grid_shape.contains(&0) {
            return Err(Error::Invalid(
                "spatial tile and grid dimensions must be positive".into(),
            ));
        }
        let mut axes = BTreeSet::new();
        for coordinate in &spatial.coordinates {
            if coordinate.axis.is_empty() || !axes.insert(coordinate.axis.as_str()) {
                return Err(Error::Invalid(
                    "spatial coordinate axes must be nonempty and unique".into(),
                ));
            }
            let column = self
                .columns
                .iter()
                .find(|column| column.name == coordinate.column)
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "coordinate column {:?} is absent",
                        coordinate.column
                    ))
                })?;
            if column.role != ColumnRole::Coordinate {
                return Err(Error::Invalid(format!(
                    "coordinate column {:?} does not have coordinate role",
                    coordinate.column
                )));
            }
        }
        if spatial.tile_order != "row_major"
            || spatial.within_tile_order != "lexicographic_coordinates_then_identity"
        {
            return Err(Error::Invalid(
                "version 1 supports row-major tiles and lexicographic coordinate/identity row order"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn tile_count(&self) -> Result<u64> {
        self.spatial_index
            .grid_shape
            .iter()
            .try_fold(1u64, |product, value| product.checked_mul(*value))
            .ok_or_else(|| Error::Invalid("spatial grid is too large".into()))
    }
}

fn validate_component(value: &str, what: &str) -> Result<()> {
    if value.is_empty() || value == "." || value == ".." || value.contains(['/', '\\']) {
        return Err(Error::Invalid(format!(
            "{what} name {value:?} is not one safe path component"
        )));
    }
    Ok(())
}

struct StoredArray {
    array: Array,
    dtype: DType,
    len: u64,
    chunk: u64,
    written: Mutex<Vec<Range<u64>>>,
}

impl StoredArray {
    fn reserve(&self, range: Range<u64>) -> Result<()> {
        if range.start > range.end || range.end > self.len {
            return Err(Error::Invalid(format!(
                "write range {:?} is outside array length {}",
                range, self.len
            )));
        }
        if range.start != range.end
            && (range.start % self.chunk != 0
                || (range.end != self.len && range.end % self.chunk != 0))
        {
            return Err(Error::Invalid(format!(
                "write range {:?} does not cover complete {}-row chunks",
                range, self.chunk
            )));
        }
        let mut written = self
            .written
            .lock()
            .map_err(|_| Error::Storage("object-table write coverage lock was poisoned".into()))?;
        if written
            .iter()
            .any(|old| old.start < range.end && range.start < old.end)
        {
            return Err(Error::Invalid(format!(
                "write range {:?} overlaps an earlier write",
                range
            )));
        }
        written.push(range);
        Ok(())
    }

    fn release(&self, range: &Range<u64>) {
        if let Ok(mut written) = self.written.lock() {
            if let Some(index) = written.iter().position(|old| old == range) {
                written.swap_remove(index);
            }
        }
    }

    fn complete(&self) -> Result<bool> {
        if self.len == 0 {
            return Ok(true);
        }
        let mut written = self
            .written
            .lock()
            .map_err(|_| Error::Storage("object-table coverage lock was poisoned".into()))?
            .clone();
        written.sort_by_key(|range| range.start);
        let mut at = 0;
        for range in written {
            if range.start != at {
                return Ok(false);
            }
            at = range.end;
        }
        Ok(at == self.len)
    }
}

/// Incremental writer for one native table.
///
/// Column writes must cover complete Zarr chunks, except for the final short
/// chunk. Distinct chunks may be written from different threads. `finish`
/// refuses missing or overlapping ranges and publishes the root metadata last.
pub struct TableWriter {
    root: PathBuf,
    spec: TableSpec,
    arrays: BTreeMap<String, Arc<StoredArray>>,
}

impl TableWriter {
    pub fn create(root: impl Into<PathBuf>, spec: TableSpec) -> Result<Self> {
        spec.validate()?;
        let root = root.into();
        if root.join("zarr.json").exists() {
            return Err(Error::Invalid(format!(
                "{} is already a published Zarr node",
                root.display()
            )));
        }
        fs::create_dir_all(&root)?;
        for group in [
            "columns",
            "index",
            "index/spatial",
            "index/spatial/counts",
            "index/identity",
        ] {
            write_group(&root.join(group))?;
        }
        let store = Arc::new(FilesystemStore::new(&root).map_err(Error::storage)?);
        let mut arrays = BTreeMap::new();
        for column in &spec.columns {
            let path = format!("/columns/{}", column.name);
            let attributes = serde_json::to_value(column)?
                .as_object()
                .cloned()
                .ok_or_else(|| Error::Invalid("column metadata is not an object".into()))?;
            arrays.insert(
                column.name.clone(),
                Arc::new(create_array(
                    &store,
                    &path,
                    spec.row_count,
                    spec.row_chunk,
                    column.dtype,
                    attributes,
                )?),
            );
        }
        arrays.insert(
            "@identity_ids".into(),
            Arc::new(create_array(
                &store,
                "/index/identity/ids",
                spec.row_count,
                spec.row_chunk,
                DType::U64,
                JsonMap::new(),
            )?),
        );
        arrays.insert(
            "@identity_rows".into(),
            Arc::new(create_array(
                &store,
                "/index/identity/rows",
                spec.row_count,
                spec.row_chunk,
                DType::U64,
                JsonMap::new(),
            )?),
        );
        create_grid_array(
            &store,
            "/index/spatial/tile_start",
            &spec.spatial_index.grid_shape,
            DType::U64,
        )?;
        create_grid_array(
            &store,
            "/index/spatial/tile_count",
            &spec.spatial_index.grid_shape,
            DType::U64,
        )?;
        for (level, shape) in occupancy_shapes(&spec.spatial_index.grid_shape)
            .into_iter()
            .enumerate()
        {
            create_grid_array(
                &store,
                &format!("/index/spatial/counts/{level}"),
                &shape,
                DType::U64,
            )?;
        }
        Ok(Self { root, spec, arrays })
    }

    pub fn spec(&self) -> &TableSpec {
        &self.spec
    }

    pub fn write_u8(&self, column: &str, start: u64, values: &[u8]) -> Result<()> {
        self.write(column, start, values, DType::U8)
    }
    pub fn write_u16(&self, column: &str, start: u64, values: &[u16]) -> Result<()> {
        self.write(column, start, values, DType::U16)
    }
    pub fn write_u32(&self, column: &str, start: u64, values: &[u32]) -> Result<()> {
        self.write(column, start, values, DType::U32)
    }
    pub fn write_u64(&self, column: &str, start: u64, values: &[u64]) -> Result<()> {
        self.write(column, start, values, DType::U64)
    }
    pub fn write_i32(&self, column: &str, start: u64, values: &[i32]) -> Result<()> {
        self.write(column, start, values, DType::I32)
    }
    pub fn write_i64(&self, column: &str, start: u64, values: &[i64]) -> Result<()> {
        self.write(column, start, values, DType::I64)
    }
    pub fn write_f32(&self, column: &str, start: u64, values: &[f32]) -> Result<()> {
        self.write(column, start, values, DType::F32)
    }
    pub fn write_f64(&self, column: &str, start: u64, values: &[f64]) -> Result<()> {
        self.write(column, start, values, DType::F64)
    }

    pub fn write_identity_index(&self, start: u64, ids: &[u64], rows: &[u64]) -> Result<()> {
        if ids.len() != rows.len() {
            return Err(Error::Invalid(
                "identity ids and rows must have the same length".into(),
            ));
        }
        self.write("@identity_ids", start, ids, DType::U64)?;
        if let Err(error) = self.write("@identity_rows", start, rows, DType::U64) {
            return Err(error);
        }
        Ok(())
    }

    pub fn write_spatial_index(&self, starts: &[u64], counts: &[u64]) -> Result<()> {
        let expected = usize::try_from(self.spec.tile_count()?)
            .map_err(|_| Error::Invalid("spatial grid does not fit this platform".into()))?;
        if starts.len() != expected || counts.len() != expected {
            return Err(Error::Invalid(format!(
                "spatial index needs {expected} starts and counts"
            )));
        }
        let store = Arc::new(FilesystemStore::new(&self.root).map_err(Error::storage)?);
        let subset = ArraySubset::new_with_shape(self.spec.spatial_index.grid_shape.clone());
        let starts_array =
            Array::open(store.clone(), "/index/spatial/tile_start").map_err(Error::storage)?;
        let counts_array =
            Array::open(store, "/index/spatial/tile_count").map_err(Error::storage)?;
        starts_array
            .store_array_subset(&subset, starts)
            .map_err(Error::storage)?;
        counts_array
            .store_array_subset(&subset, counts)
            .map_err(Error::storage)?;
        let mut previous_shape = self.spec.spatial_index.grid_shape.clone();
        let mut previous = counts.to_vec();
        for (level, shape) in occupancy_shapes(&previous_shape).into_iter().enumerate() {
            let values = if level == 0 {
                previous.clone()
            } else {
                sum_occupancy(&previous, &previous_shape, &shape)?
            };
            let array = Array::open(
                Arc::new(FilesystemStore::new(&self.root).map_err(Error::storage)?),
                &format!("/index/spatial/counts/{level}"),
            )
            .map_err(Error::storage)?;
            let level_subset = ArraySubset::new_with_shape(shape.clone());
            array
                .store_array_subset(&level_subset, &values)
                .map_err(Error::storage)?;
            previous = values;
            previous_shape = shape;
        }
        Ok(())
    }

    fn write<T: Element>(
        &self,
        column: &str,
        start: u64,
        values: &[T],
        dtype: DType,
    ) -> Result<()> {
        let stored = self
            .arrays
            .get(column)
            .ok_or_else(|| Error::Invalid(format!("unknown column {column:?}")))?;
        if stored.dtype != dtype {
            return Err(Error::Invalid(format!(
                "column {column:?} is {:?}, not {:?}",
                stored.dtype, dtype
            )));
        }
        let length = u64::try_from(values.len())
            .map_err(|_| Error::Invalid("write length does not fit u64".into()))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| Error::Invalid("write range overflows u64".into()))?;
        let range = start..end;
        stored.reserve(range.clone())?;
        let subset =
            ArraySubset::new_with_start_shape(vec![start], vec![length]).map_err(Error::storage)?;
        if let Err(error) = stored.array.store_array_subset(&subset, values) {
            stored.release(&range);
            return Err(Error::storage(error));
        }
        Ok(())
    }

    pub fn finish(self) -> Result<TableReader> {
        for (name, array) in &self.arrays {
            if !array.complete()? {
                return Err(Error::Invalid(format!(
                    "array {name:?} is not completely written"
                )));
            }
        }
        validate_index_ranges(&self.root, &self.spec)?;
        let metadata = json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": { METADATA_KEY: self.spec }
        });
        write_json_atomic(&self.root.join("zarr.json"), &metadata)?;
        TableReader::open(self.root)
    }
}

fn create_array(
    store: &Arc<Store>,
    path: &str,
    len: u64,
    chunk: u64,
    dtype: DType,
    attributes: JsonMap<String, Json>,
) -> Result<StoredArray> {
    let chunk = chunk.min(len.max(1));
    let mut builder = ArrayBuilder::new(vec![len], vec![chunk], dtype.zarr(), dtype.fill());
    builder.attributes(attributes);
    let gzip = GzipCodec::new(1).map_err(Error::storage)?;
    builder.bytes_to_bytes_codecs(vec![Arc::new(gzip) as Arc<dyn BytesToBytesCodecTraits>]);
    let array = builder.build(store.clone(), path).map_err(Error::storage)?;
    array.store_metadata().map_err(Error::storage)?;
    Ok(StoredArray {
        array,
        dtype,
        len,
        chunk,
        written: Mutex::new(Vec::new()),
    })
}

fn create_grid_array(store: &Arc<Store>, path: &str, shape: &[u64], dtype: DType) -> Result<()> {
    let mut builder = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), dtype.zarr(), dtype.fill());
    let gzip = GzipCodec::new(1).map_err(Error::storage)?;
    builder.bytes_to_bytes_codecs(vec![Arc::new(gzip) as Arc<dyn BytesToBytesCodecTraits>]);
    let array = builder.build(store.clone(), path).map_err(Error::storage)?;
    array.store_metadata().map_err(Error::storage)
}

fn validate_index_ranges(root: &Path, spec: &TableSpec) -> Result<()> {
    let store = Arc::new(FilesystemStore::new(root).map_err(Error::storage)?);
    let subset = ArraySubset::new_with_shape(spec.spatial_index.grid_shape.clone());
    let starts: Vec<u64> = Array::open(store.clone(), "/index/spatial/tile_start")
        .map_err(Error::storage)?
        .retrieve_array_subset(&subset)
        .map_err(Error::storage)?;
    let counts: Vec<u64> = Array::open(store, "/index/spatial/tile_count")
        .map_err(Error::storage)?
        .retrieve_array_subset(&subset)
        .map_err(Error::storage)?;
    let mut expected = 0u64;
    for (tile, (&start, &count)) in starts.iter().zip(&counts).enumerate() {
        if start != expected {
            return Err(Error::Invalid(format!(
                "spatial tile {tile} starts at row {start}, expected {expected}"
            )));
        }
        expected = expected
            .checked_add(count)
            .ok_or_else(|| Error::Invalid("spatial row ranges overflow u64".into()))?;
    }
    if expected != spec.row_count {
        return Err(Error::Invalid(format!(
            "spatial index covers {expected} rows, table has {}",
            spec.row_count
        )));
    }
    Ok(())
}

/// Partial reader for a completed native table.
pub struct TableReader {
    root: PathBuf,
    spec: TableSpec,
    store: Arc<Store>,
}

impl TableReader {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let metadata: Json = serde_json::from_slice(&fs::read(root.join("zarr.json"))?)?;
        if metadata.get("zarr_format").and_then(Json::as_u64) != Some(3)
            || metadata.get("node_type").and_then(Json::as_str) != Some("group")
        {
            return Err(Error::Invalid(format!(
                "{} is not a Zarr v3 group",
                root.display()
            )));
        }
        let spec: TableSpec = serde_json::from_value(
            metadata
                .pointer(&format!("/attributes/{METADATA_KEY}"))
                .cloned()
                .ok_or_else(|| Error::Invalid("group has no object_table metadata".into()))?,
        )?;
        spec.validate()?;
        let store = Arc::new(FilesystemStore::new(&root).map_err(Error::storage)?);
        for column in &spec.columns {
            let array = Array::open(store.clone(), &format!("/columns/{}", column.name))
                .map_err(Error::storage)?;
            if array.shape() != [spec.row_count] || *array.data_type() != column.dtype.zarr() {
                return Err(Error::Invalid(format!(
                    "column {:?} does not match its declared shape or type",
                    column.name
                )));
            }
        }
        Ok(Self { root, spec, store })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn spec(&self) -> &TableSpec {
        &self.spec
    }

    pub fn read_u8(&self, column: &str, range: Range<u64>) -> Result<Vec<u8>> {
        self.read(column, range, DType::U8)
    }
    pub fn read_u16(&self, column: &str, range: Range<u64>) -> Result<Vec<u16>> {
        self.read(column, range, DType::U16)
    }
    pub fn read_u32(&self, column: &str, range: Range<u64>) -> Result<Vec<u32>> {
        self.read(column, range, DType::U32)
    }
    pub fn read_u64(&self, column: &str, range: Range<u64>) -> Result<Vec<u64>> {
        self.read(column, range, DType::U64)
    }
    pub fn read_i32(&self, column: &str, range: Range<u64>) -> Result<Vec<i32>> {
        self.read(column, range, DType::I32)
    }
    pub fn read_i64(&self, column: &str, range: Range<u64>) -> Result<Vec<i64>> {
        self.read(column, range, DType::I64)
    }
    pub fn read_f32(&self, column: &str, range: Range<u64>) -> Result<Vec<f32>> {
        self.read(column, range, DType::F32)
    }
    pub fn read_f64(&self, column: &str, range: Range<u64>) -> Result<Vec<f64>> {
        self.read(column, range, DType::F64)
    }

    pub fn read_identity_index(&self, range: Range<u64>) -> Result<(Vec<u64>, Vec<u64>)> {
        Ok((
            self.read_path("/index/identity/ids", range.clone(), DType::U64)?,
            self.read_path("/index/identity/rows", range, DType::U64)?,
        ))
    }

    pub fn read_spatial_index(&self) -> Result<(Vec<u64>, Vec<u64>)> {
        self.read_spatial_index_region(
            &vec![0; self.spec.spatial_index.grid_shape.len()],
            &self.spec.spatial_index.grid_shape,
        )
    }

    /// Read only the spatial-index cells intersecting a tile-grid region.
    pub fn read_spatial_index_region(
        &self,
        start: &[u64],
        shape: &[u64],
    ) -> Result<(Vec<u64>, Vec<u64>)> {
        if start.len() != self.spec.spatial_index.grid_shape.len()
            || shape.len() != self.spec.spatial_index.grid_shape.len()
        {
            return Err(Error::Invalid(format!(
                "spatial index has rank {}, got start rank {} and shape rank {}",
                self.spec.spatial_index.grid_shape.len(),
                start.len(),
                shape.len()
            )));
        }
        for axis in 0..start.len() {
            if start[axis]
                .checked_add(shape[axis])
                .is_none_or(|end| end > self.spec.spatial_index.grid_shape[axis])
            {
                return Err(Error::Invalid(format!(
                    "spatial index region exceeds grid shape {:?} on axis {axis}",
                    self.spec.spatial_index.grid_shape
                )));
            }
        }
        let subset = ArraySubset::new_with_start_shape(start.to_vec(), shape.to_vec())
            .map_err(Error::storage)?;
        let starts = Array::open(self.store.clone(), "/index/spatial/tile_start")
            .map_err(Error::storage)?
            .retrieve_array_subset(&subset)
            .map_err(Error::storage)?;
        let counts = Array::open(self.store.clone(), "/index/spatial/tile_count")
            .map_err(Error::storage)?
            .retrieve_array_subset(&subset)
            .map_err(Error::storage)?;
        Ok((starts, counts))
    }

    pub fn occupancy_shapes(&self) -> Vec<Vec<u64>> {
        occupancy_shapes(&self.spec.spatial_index.grid_shape)
    }

    /// Read a region from one level of the summed occupancy pyramid.
    pub fn read_occupancy(&self, level: usize, start: &[u64], shape: &[u64]) -> Result<Vec<u64>> {
        let levels = self.occupancy_shapes();
        let level_shape = levels.get(level).ok_or_else(|| {
            Error::Invalid(format!(
                "occupancy level {level} is absent; table has {} levels",
                levels.len()
            ))
        })?;
        if start.len() != level_shape.len() || shape.len() != level_shape.len() {
            return Err(Error::Invalid(format!(
                "occupancy level has rank {}, got start rank {} and shape rank {}",
                level_shape.len(),
                start.len(),
                shape.len()
            )));
        }
        for axis in 0..start.len() {
            if start[axis]
                .checked_add(shape[axis])
                .is_none_or(|end| end > level_shape[axis])
            {
                return Err(Error::Invalid(format!(
                    "occupancy region exceeds level shape {level_shape:?} on axis {axis}"
                )));
            }
        }
        let subset = ArraySubset::new_with_start_shape(start.to_vec(), shape.to_vec())
            .map_err(Error::storage)?;
        Array::open(
            self.store.clone(),
            &format!("/index/spatial/counts/{level}"),
        )
        .map_err(Error::storage)?
        .retrieve_array_subset(&subset)
        .map_err(Error::storage)
    }

    fn read<T: ElementOwned>(
        &self,
        column: &str,
        range: Range<u64>,
        dtype: DType,
    ) -> Result<Vec<T>> {
        let declared = self
            .spec
            .columns
            .iter()
            .find(|candidate| candidate.name == column)
            .ok_or_else(|| Error::Invalid(format!("unknown column {column:?}")))?;
        if declared.dtype != dtype {
            return Err(Error::Invalid(format!(
                "column {column:?} is {:?}, not {:?}",
                declared.dtype, dtype
            )));
        }
        self.read_path(&format!("/columns/{column}"), range, dtype)
    }

    fn read_path<T: ElementOwned>(
        &self,
        path: &str,
        range: Range<u64>,
        dtype: DType,
    ) -> Result<Vec<T>> {
        if range.start > range.end || range.end > self.spec.row_count {
            return Err(Error::Invalid(format!(
                "read range {:?} is outside row count {}",
                range, self.spec.row_count
            )));
        }
        let array = Array::open(self.store.clone(), path).map_err(Error::storage)?;
        if *array.data_type() != dtype.zarr() {
            return Err(Error::Invalid(format!("array {path:?} has the wrong type")));
        }
        let subset =
            ArraySubset::new_with_start_shape(vec![range.start], vec![range.end - range.start])
                .map_err(Error::storage)?;
        array.retrieve_array_subset(&subset).map_err(Error::storage)
    }
}

fn write_group(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    fs::write(
        path.join("zarr.json"),
        serde_json::to_vec_pretty(&json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {}
        }))?,
    )?;
    Ok(())
}

fn occupancy_shapes(base: &[u64]) -> Vec<Vec<u64>> {
    let mut levels = vec![base.to_vec()];
    while levels
        .last()
        .is_some_and(|shape| shape.iter().any(|&n| n > 1))
    {
        levels.push(
            levels
                .last()
                .expect("level zero exists")
                .iter()
                .map(|&n| n.div_ceil(2))
                .collect(),
        );
    }
    levels
}

fn sum_occupancy(previous: &[u64], old_shape: &[u64], new_shape: &[u64]) -> Result<Vec<u64>> {
    let old_len = element_count(old_shape)?;
    if previous.len() != old_len {
        return Err(Error::Invalid(
            "occupancy values do not match their shape".into(),
        ));
    }
    let new_len = element_count(new_shape)?;
    let mut out = vec![0u64; new_len];
    let mut coordinate = vec![0u64; old_shape.len()];
    for (linear, &value) in previous.iter().enumerate() {
        unravel(linear as u64, old_shape, &mut coordinate);
        for value in &mut coordinate {
            *value /= 2;
        }
        let target = ravel(&coordinate, new_shape)?;
        out[target] = out[target]
            .checked_add(value)
            .ok_or_else(|| Error::Invalid("occupancy count overflows uint64".into()))?;
    }
    Ok(out)
}

fn element_count(shape: &[u64]) -> Result<usize> {
    let count = shape
        .iter()
        .try_fold(1u64, |product, &n| product.checked_mul(n));
    usize::try_from(count.ok_or_else(|| Error::Invalid("array shape is too large".into()))?)
        .map_err(|_| Error::Invalid("array shape does not fit this platform".into()))
}

fn unravel(mut linear: u64, shape: &[u64], coordinate: &mut [u64]) {
    for axis in (0..shape.len()).rev() {
        coordinate[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
}

fn ravel(coordinate: &[u64], shape: &[u64]) -> Result<usize> {
    let linear = coordinate
        .iter()
        .zip(shape)
        .try_fold(0u64, |linear, (&at, &length)| {
            linear.checked_mul(length)?.checked_add(at)
        })
        .ok_or_else(|| Error::Invalid("array coordinate overflows uint64".into()))?;
    usize::try_from(linear)
        .map_err(|_| Error::Invalid("array coordinate does not fit this platform".into()))
}

fn write_json_atomic(path: &Path, value: &Json) -> Result<()> {
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
