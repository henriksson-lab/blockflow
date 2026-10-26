//! Disk-backed, locally retained physical features using the existing table
//! encoding. One bounded table per voxel cell; a sorted cell directory streams
//! occupied cells without a resident organ-wide point table or directory.
use crate::{
    budget::MemoryBudget,
    error::{Error, Result},
    region::Region,
    table::{encoded_header_bytes, encoded_row_bytes, Column, Layout, Schema, Table, Value},
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
const META_LIMIT: u64 = 16384;
/// Physical page layout. Feature ordering, bindings and retention are identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PageLayout {
    #[default]
    Files,
    /// Version 2: append-only page payload plus fixed-width offset directory.
    Packed,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    pub shape_zyx: [usize; 3],
    pub cell_zyx: [usize; 3],
    pub origin_xyz: [f64; 3],
    pub spacing_xyz: [f64; 3],
    pub units: String,
    /// Maximum point displacement from its integer anchor, per axis in voxels.
    pub max_offset_voxels: f64,
}
impl Geometry {
    pub fn validate(&self) -> Result<()> {
        if self.shape_zyx.contains(&0)
            || self.cell_zyx.contains(&0)
            || self.shape_zyx.iter().any(|&v| v as f64 > 2f64.powi(53))
            || self.origin_xyz.iter().any(|v| !v.is_finite())
            || self.spacing_xyz.iter().any(|v| !v.is_finite() || *v <= 0.)
            || self.units.trim().is_empty()
            || self.units.len() > 128
            || !self.max_offset_voxels.is_finite()
            || !(0.0..=1.).contains(&self.max_offset_voxels)
        {
            return Err(Error::invalid("invalid physical feature geometry"));
        }
        for i in 0..3 {
            let p = self.origin_xyz[i]
                + (self.shape_zyx[2 - i] as f64 + self.max_offset_voxels) * self.spacing_xyz[i];
            if !p.is_finite() {
                return Err(Error::invalid("feature world bounds overflow"));
            }
        }
        Ok(())
    }
    pub fn cells_zyx(&self) -> [usize; 3] {
        std::array::from_fn(|i| self.shape_zyx[i].div_ceil(self.cell_zyx[i]))
    }
    pub fn cell_of(&self, anchor: [usize; 3]) -> Result<[usize; 3]> {
        if (0..3).any(|i| anchor[i] >= self.shape_zyx[i]) {
            return Err(Error::invalid("feature anchor outside image"));
        }
        Ok(std::array::from_fn(|i| anchor[i] / self.cell_zyx[i]))
    }
    pub fn cell_region(&self, cell: [usize; 3]) -> Result<Region> {
        if (0..3).any(|i| cell[i] >= self.cells_zyx()[i]) {
            return Err(Error::invalid("feature cell outside image"));
        }
        let start: [usize; 3] = std::array::from_fn(|i| cell[i] * self.cell_zyx[i]);
        let shape: [usize; 3] =
            std::array::from_fn(|i| self.cell_zyx[i].min(self.shape_zyx[i] - start[i]));
        Ok(Region::new(&start, &shape))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    pub at_zyx: [usize; 3],
    pub position_xyz: [f64; 3],
    pub weight: f64,
    pub scale: f64,
}
fn anchor_roundoff_voxels(origin: f64, position: f64, spacing: f64, extent: f64) -> Result<f64> {
    let magnitude = extent
        .abs()
        .max(1.)
        .max((origin / spacing).abs())
        .max((position / spacing).abs());
    let roundoff = 16. * f64::EPSILON * magnitude;
    // Refuse an unresolvable frame instead of making roundoff a substantial
    // extra displacement allowance.
    if !roundoff.is_finite() || roundoff >= 0.25 {
        return Err(Error::invalid(
            "feature world coordinates lack subvoxel precision",
        ));
    }
    Ok(roundoff)
}
impl Feature {
    fn checked(mut self, g: &Geometry, cell: [usize; 3]) -> Result<Self> {
        if g.cell_of(self.at_zyx)? != cell
            || !self.weight.is_finite()
            || !self.scale.is_finite()
            || self.scale <= 0.
            || self.position_xyz.iter().any(|v| !v.is_finite())
        {
            return Err(Error::invalid("invalid feature or wrong cell ownership"));
        }
        for i in 0..3 {
            let q = (self.position_xyz[i] - g.origin_xyz[i]) / g.spacing_xyz[i];
            let delta = (q - self.at_zyx[2 - i] as f64).abs();
            // World-coordinate construction rounds at the magnitude of the
            // origin/position, not only at the small local voxel coordinate.
            // Include both before testing a boundary such as +/- half a voxel.
            let roundoff_voxels = anchor_roundoff_voxels(
                g.origin_xyz[i],
                self.position_xyz[i],
                g.spacing_xyz[i],
                q.abs().max(self.at_zyx[2 - i] as f64),
            )?;
            if !delta.is_finite() || delta > g.max_offset_voxels + roundoff_voxels {
                return Err(Error::invalid(
                    "feature exceeds declared anchor displacement",
                ));
            }
            if self.position_xyz[i] == 0. {
                self.position_xyz[i] = 0.;
            }
        }
        if self.weight == 0. {
            self.weight = 0.;
        }
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FeatureKey {
    pub cell_zyx: [usize; 3],
    pub row: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocatedFeature {
    pub key: FeatureKey,
    pub feature: Feature,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub version: u32,
    pub sealed: bool,
    /// Caller-supplied digest of immutable source/version, channel/time/level,
    /// geometry, detector and retention policy. Not a proof of immutability.
    pub binding_sha256: String,
    pub geometry: Geometry,
    pub per_cell_limit: usize,
    pub submitted_points: u64,
    pub retained_points: u64,
    pub occupied_cells: u64,
    pub full_cells: u64,
    pub retention_events: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Limits {
    pub max_cell_visits: u64,
    pub max_read_calls: u64,
    pub max_write_calls: u64,
    pub max_bytes_read: u64,
    pub max_bytes_written: u64,
    pub max_work_units: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_cell_visits: 1_000_000,
            max_read_calls: 1_000_000,
            max_write_calls: 1_000_000,
            max_bytes_read: 8 * 1024 * 1024 * 1024,
            max_bytes_written: 8 * 1024 * 1024 * 1024,
            max_work_units: 100_000_000,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Stats {
    pub cell_visits: u64,
    pub read_calls: u64,
    pub directory_read_calls: u64,
    pub directory_bytes_read: u64,
    pub directory_probes: u64,
    pub page_read_calls: u64,
    pub page_bytes_read: u64,
    pub page_decodes: u64,
    pub write_calls: u64,
    pub charged_bytes_read: u64,
    pub charged_bytes_written: u64,
    pub work_units: u64,
    pub peak_planned_working_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    Cells,
    Reads,
    Writes,
    ReadBytes,
    WriteBytes,
    Work,
    Memory,
}
#[derive(Debug)]
pub enum Outcome<T> {
    Complete(T),
    Stopped(Stop),
}
impl<T> Outcome<T> {
    pub fn into_complete(self) -> Option<T> {
        match self {
            Self::Complete(v) => Some(v),
            Self::Stopped(_) => None,
        }
    }
}
/// One cumulative ledger, reusable across calls. Stop is sticky; a partially
/// completed operation must not be interpreted as an empty neighborhood.
pub struct Budget {
    pub limits: Limits,
    pub stats: Stats,
    pub stop: Option<Stop>,
}
impl Budget {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            stats: Stats::default(),
            stop: None,
        }
    }
    fn cell(&mut self) -> bool {
        if self.stop.is_some() {
            return false;
        }
        if self.stats.cell_visits >= self.limits.max_cell_visits {
            self.stop = Some(Stop::Cells);
            return false;
        }
        self.stats.cell_visits += 1;
        true
    }
    fn work(&mut self, n: u64) -> bool {
        if self.stop.is_some() {
            return false;
        }
        if n > self
            .limits
            .max_work_units
            .saturating_sub(self.stats.work_units)
        {
            self.stop = Some(Stop::Work);
            return false;
        }
        self.stats.work_units += n;
        true
    }
    fn read(&mut self, bytes: u64) -> bool {
        if self.stop.is_some() {
            return false;
        }
        if self.stats.read_calls >= self.limits.max_read_calls {
            self.stop = Some(Stop::Reads);
            return false;
        }
        if bytes
            > self
                .limits
                .max_bytes_read
                .saturating_sub(self.stats.charged_bytes_read)
        {
            self.stop = Some(Stop::ReadBytes);
            return false;
        }
        self.stats.read_calls += 1;
        self.stats.charged_bytes_read += bytes;
        true
    }
    fn directory_read(&mut self, bytes: u64) -> bool {
        if !self.read(bytes) {
            return false;
        }
        self.stats.directory_read_calls += 1;
        self.stats.directory_bytes_read += bytes;
        true
    }
    fn page_read(&mut self, bytes: u64) -> bool {
        if !self.read(bytes) {
            return false;
        }
        self.stats.page_read_calls += 1;
        self.stats.page_bytes_read += bytes;
        true
    }
    fn write(&mut self, bytes: u64) -> bool {
        if self.stop.is_some() {
            return false;
        }
        if self.stats.write_calls >= self.limits.max_write_calls {
            self.stop = Some(Stop::Writes);
            return false;
        }
        if bytes
            > self
                .limits
                .max_bytes_written
                .saturating_sub(self.stats.charged_bytes_written)
        {
            self.stop = Some(Stop::WriteBytes);
            return false;
        }
        self.stats.write_calls += 1;
        self.stats.charged_bytes_written += bytes;
        true
    }
    fn stopped<T>(&self) -> Outcome<T> {
        Outcome::Stopped(self.stop.expect("stopped budget"))
    }
}
fn schema() -> Schema {
    Schema::new(vec![
        Column::f64("x"),
        Column::f64("y"),
        Column::f64("z"),
        Column::f64("weight"),
        Column::f64("scale"),
    ])
    .expect("fixed feature schema")
}
fn values(p: &Feature) -> [Value; 5] {
    [
        Value::F64(p.position_xyz[0]),
        Value::F64(p.position_xyz[1]),
        Value::F64(p.position_xyz[2]),
        Value::F64(p.weight),
        Value::F64(p.scale),
    ]
}
fn rank(a: &Feature, b: &Feature) -> Ordering {
    b.weight
        .total_cmp(&a.weight)
        .then(a.at_zyx.cmp(&b.at_zyx))
        .then(a.scale.total_cmp(&b.scale))
        .then_with(|| {
            a.position_xyz
                .iter()
                .zip(b.position_xyz)
                .map(|(a, b)| a.total_cmp(&b))
                .find(|o| !o.is_eq())
                .unwrap_or(Ordering::Equal)
        })
}
fn atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid("index path has no parent"))?;
    fs::create_dir_all(parent).map_err(Error::backend)?;
    let temp = path.with_extension("partial");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(Error::backend)?;
    file.write_all(bytes).map_err(Error::backend)?;
    file.sync_all().map_err(Error::backend)?;
    fs::rename(&temp, path).map_err(Error::backend)
}
fn page(root: &Path, key: [usize; 3]) -> PathBuf {
    root.join("pages")
        .join(key[0].to_string())
        .join(key[1].to_string())
        .join(format!("{}.rows", key[2]))
}
fn plan(cap: usize, extra: usize) -> Result<u64> {
    // Conservative payload allowance for table decoding/sorting/encoding and
    // feature vectors, not allocator/process RSS or unrelated caller buffers.
    (cap as u64)
        .checked_add(extra as u64)
        .and_then(|n| n.checked_mul(1024))
        .and_then(|n| n.checked_add(65536))
        .ok_or_else(|| Error::invalid("feature index working size overflow"))
}
fn lease(memory: &MemoryBudget, bytes: u64, b: &mut Budget) -> Option<crate::budget::Lease> {
    if b.stop.is_some() {
        return None;
    }
    let hold = memory.try_acquire(bytes);
    if hold.is_none() {
        b.stop = Some(Stop::Memory);
    } else {
        b.stats.peak_planned_working_bytes =
            b.stats.peak_planned_working_bytes.max(memory.in_use());
    }
    hold
}
fn check_meta(m: &Metadata) -> Result<()> {
    m.geometry.validate()?;
    if !matches!(m.version, 1 | 2)
        || m.per_cell_limit == 0
        || m.per_cell_limit > 4096
        || m.binding_sha256.len() != 64
        || !m.binding_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(Error::invalid("invalid feature index metadata"));
    }
    Ok(())
}
pub struct Writer {
    root: PathBuf,
    meta: Metadata,
    directory: File,
    packed: Option<PackedOutput>,
    last: Option<[usize; 3]>,
    poisoned: bool,
    memory: MemoryBudget,
}
struct PackedOutput {
    pages: File,
    offsets: File,
    bytes: u64,
}
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .map_err(Error::backend)?
        .sync_all()
        .map_err(Error::backend)
}
impl Writer {
    pub fn create(
        root: impl Into<PathBuf>,
        geometry: Geometry,
        binding_sha256: String,
        per_cell_limit: usize,
        memory: MemoryBudget,
    ) -> Result<Self> {
        Self::create_with_layout(
            root,
            geometry,
            binding_sha256,
            per_cell_limit,
            memory,
            PageLayout::Files,
        )
    }
    /// Packed publication synchronizes payload/offset/key files before sealing.
    /// Existing create() keeps its version-1 layout and publication behavior.
    pub fn create_with_layout(
        root: impl Into<PathBuf>,
        geometry: Geometry,
        binding_sha256: String,
        per_cell_limit: usize,
        memory: MemoryBudget,
        layout: PageLayout,
    ) -> Result<Self> {
        let meta = Metadata {
            version: if layout == PageLayout::Packed { 2 } else { 1 },
            sealed: false,
            binding_sha256,
            geometry,
            per_cell_limit,
            submitted_points: 0,
            retained_points: 0,
            occupied_cells: 0,
            full_cells: 0,
            retention_events: 0,
        };
        check_meta(&meta)?;
        let root = root.into();
        fs::create_dir(&root).map_err(Error::backend)?;
        atomic(
            &root.join("manifest.json"),
            &serde_json::to_vec(&meta).map_err(Error::backend)?,
        )?;
        let directory = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("cells.bin"))
            .map_err(Error::backend)?;
        let packed = if layout == PageLayout::Packed {
            Some(PackedOutput {
                pages: OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join("pages.bin"))
                    .map_err(Error::backend)?,
                offsets: OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(root.join("page_offsets.bin"))
                    .map_err(Error::backend)?,
                bytes: 0,
            })
        } else {
            None
        };
        Ok(Self {
            root,
            meta,
            directory,
            packed,
            last: None,
            poisoned: false,
            memory,
        })
    }
    pub fn metadata(&self) -> &Metadata {
        &self.meta
    }
    /// Each logical retention cell is supplied once in increasing ZYX order.
    /// Input order *within* a cell is arbitrary and may be streamed. Top-K and
    /// duplicate-anchor/scale max-weight merging are deterministic. IO chunking
    /// is independent of the fixed, versioned retention-cell policy.
    pub fn write_cell(
        &mut self,
        key: [usize; 3],
        points: impl IntoIterator<Item = Feature>,
        b: &mut Budget,
    ) -> Result<Outcome<usize>> {
        if self.poisoned {
            return Err(Error::invalid(
                "feature writer is poisoned; cannot publish partial index",
            ));
        }
        self.poisoned = true;
        let result = self.write_inner(key, points, b);
        if matches!(&result, Ok(Outcome::Complete(_))) {
            self.poisoned = false;
        }
        result
    }
    fn write_inner(
        &mut self,
        key: [usize; 3],
        points: impl IntoIterator<Item = Feature>,
        b: &mut Budget,
    ) -> Result<Outcome<usize>> {
        self.meta.geometry.cell_region(key)?;
        if self.last.is_some_and(|last| key <= last) {
            return Err(Error::invalid(
                "feature cells must be written once in increasing ZYX order",
            ));
        }
        if !b.cell() {
            return Ok(b.stopped());
        }
        let Some(_hold) = lease(&self.memory, plan(self.meta.per_cell_limit, 0)?, b) else {
            return Ok(b.stopped());
        };
        let mut kept: Vec<Feature> = Vec::with_capacity(self.meta.per_cell_limit);
        let mut submitted = 0u64;
        let mut limited = false;
        for p in points {
            if !b.work(1 + 2 * kept.len() as u64) {
                return Ok(b.stopped());
            }
            let p = p.checked(&self.meta.geometry, key)?;
            submitted = submitted
                .checked_add(1)
                .ok_or_else(|| Error::invalid("feature count overflow"))?;
            if let Some(i) = kept
                .iter()
                .position(|q| q.at_zyx == p.at_zyx && q.scale == p.scale)
            {
                if rank(&p, &kept[i]).is_lt() {
                    kept[i] = p;
                }
            } else if kept.len() < self.meta.per_cell_limit {
                kept.push(p);
            } else {
                limited = true;
                let worst = (0..kept.len())
                    .max_by(|&a, &z| rank(&kept[a], &kept[z]))
                    .expect("nonempty retained cell");
                if rank(&p, &kept[worst]).is_lt() {
                    kept[worst] = p;
                }
            }
        }
        if !kept.is_empty() {
            // Use the existing table to obtain its canonical row order, then
            // encode with the same public RowBuilder used by fragment producers.
            let mut table = Table::new(self.meta.geometry.shape_zyx, schema())?;
            for p in &kept {
                table.push(p.at_zyx, &values(p))?;
            }
            table.seal_as(Layout::Flat)?;
            let mut encoded = table.builder();
            for row in table.scan(&self.meta.geometry.cell_region(key)?)? {
                encoded.push(
                    row.at(),
                    &(0..5).map(|i| row.value(i)).collect::<Result<Vec<_>>>()?,
                )?;
            }
            let bytes = encoded.encode();
            let directory_bytes = if self.packed.is_some() { 40 } else { 24 };
            if !b.write(bytes.len() as u64 + directory_bytes) {
                return Ok(b.stopped());
            }
            if let Some(packed) = &mut self.packed {
                let end = packed
                    .bytes
                    .checked_add(bytes.len() as u64)
                    .ok_or_else(|| Error::invalid("packed feature payload overflow"))?;
                packed.pages.write_all(&bytes).map_err(Error::backend)?;
                packed
                    .offsets
                    .write_all(&packed.bytes.to_le_bytes())
                    .map_err(Error::backend)?;
                packed
                    .offsets
                    .write_all(&(bytes.len() as u64).to_le_bytes())
                    .map_err(Error::backend)?;
                packed.bytes = end;
            } else {
                atomic(&page(&self.root, key), &bytes)?;
            }
            for value in key {
                self.directory
                    .write_all(&(value as u64).to_le_bytes())
                    .map_err(Error::backend)?;
            }
            self.meta.occupied_cells = self
                .meta
                .occupied_cells
                .checked_add(1)
                .ok_or_else(|| Error::invalid("feature cell count overflow"))?;
        }
        self.meta.submitted_points = self
            .meta
            .submitted_points
            .checked_add(submitted)
            .ok_or_else(|| Error::invalid("feature count overflow"))?;
        self.meta.retained_points = self
            .meta
            .retained_points
            .checked_add(kept.len() as u64)
            .ok_or_else(|| Error::invalid("feature count overflow"))?;
        self.meta.full_cells += u64::from(kept.len() == self.meta.per_cell_limit);
        self.meta.retention_events += u64::from(limited);
        self.last = Some(key);
        Ok(Outcome::Complete(kept.len()))
    }
    pub fn finish(mut self) -> Result<Metadata> {
        if self.poisoned {
            return Err(Error::invalid("cannot seal a partial/failed feature index"));
        }
        if let Some(packed) = &self.packed {
            packed.pages.sync_all().map_err(Error::backend)?;
            packed.offsets.sync_all().map_err(Error::backend)?;
        }
        self.directory.sync_all().map_err(Error::backend)?;
        if self.packed.is_some() {
            // Persist data-file names and the new root before publishing the
            // sealed manifest. Ancestors of the supplied parent are caller-owned.
            sync_directory(&self.root)?;
            sync_directory(
                self.root
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new(".")),
            )?;
        }
        self.meta.sealed = true;
        // Publication replaces the manifest only after all pages/directory are
        // complete. No resumability or concurrent writers are claimed.
        let path = self.root.join("manifest.json");
        atomic(
            &path,
            &serde_json::to_vec(&self.meta).map_err(Error::backend)?,
        )?;
        if self.packed.is_some() {
            // If this post-rename sync fails, a complete sealed index can be
            // visible despite Err; the caller must not record successful seal.
            sync_directory(&self.root)?;
        }
        Ok(self.meta)
    }
}
#[derive(Debug)]
pub struct Neighbors {
    pub points: Vec<LocatedFeature>,
    pub truncated: bool,
    pub matching_points: u64,
    /// In-radius points before the selector; excluded points are not matches.
    pub radius_points: u64,
    /// Eligible points per partition, before each partition's retention cap.
    pub partition_counts: Vec<u64>,
}
/// Optional bounded reuse for an immutable index. Zero bytes preserves the
/// uncached behavior. Cache bytes share the caller's MemoryBudget.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReadCacheConfig {
    pub max_bytes: u64,
    pub directory_block_keys: usize,
}
impl Default for ReadCacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 0,
            directory_block_keys: 128,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReadCacheStats {
    pub directory_hits: u64,
    pub page_hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub admission_declined: u64,
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CacheKey {
    Directory(u64),
    Page([usize; 3]),
}
enum CachedData {
    Directory(Vec<u8>),
    Page(Vec<LocatedFeature>),
}
struct CachedEntry {
    data: CachedData,
    stamp: u128,
    bytes: u64,
    _lease: crate::budget::Lease,
}
#[derive(Default)]
struct CacheState {
    entries: std::collections::BTreeMap<CacheKey, CachedEntry>,
    order: std::collections::BTreeSet<(u128, CacheKey)>,
    stamp: u128,
    stats: ReadCacheStats,
}
impl CacheState {
    fn evict_one(&mut self) -> u64 {
        let Some((_, key)) = self.order.pop_first() else {
            return 0;
        };
        let entry = self.entries.remove(&key).unwrap();
        self.stats.resident_bytes -= entry.bytes;
        self.stats.evictions += 1;
        entry.bytes
    }
}
struct ReadCache {
    config: ReadCacheConfig,
    state: std::sync::Mutex<CacheState>,
    memory: MemoryBudget,
}
impl crate::budget::Reclaim for ReadCache {
    fn reclaim(&self, bytes: u64) -> u64 {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let mut freed = 0u64;
        while freed < bytes {
            let n = state.evict_one();
            if n == 0 {
                break;
            }
            freed = freed.saturating_add(n);
        }
        freed
    }
}
impl ReadCache {
    fn new(config: ReadCacheConfig, memory: MemoryBudget) -> std::sync::Arc<Self> {
        let cache = std::sync::Arc::new(Self {
            config,
            state: Default::default(),
            memory: memory.clone(),
        });
        let reclaimer: std::sync::Arc<dyn crate::budget::Reclaim> = cache.clone();
        memory.register_reclaimer(std::sync::Arc::downgrade(&reclaimer));
        cache
    }
    fn directory(&self, block: u64, offset: usize) -> Option<[usize; 3]> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.stamp += 1;
        let stamp = state.stamp;
        let key = CacheKey::Directory(block);
        let mut previous = None;
        let result = state.entries.get_mut(&key).and_then(|e| {
            previous = Some(e.stamp);
            e.stamp = stamp;
            let CachedData::Directory(bytes) = &e.data else {
                return None;
            };
            decode_key(&bytes[offset * 24..offset * 24 + 24]).ok()
        });
        if let Some(old) = previous {
            state.order.remove(&(old, key));
            state.order.insert((stamp, key));
        }
        if result.is_some() {
            state.stats.directory_hits += 1;
        } else {
            state.stats.misses += 1;
        }
        result
    }
    fn page(&self, key: [usize; 3]) -> Option<Vec<LocatedFeature>> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.stamp += 1;
        let stamp = state.stamp;
        let key = CacheKey::Page(key);
        let mut previous = None;
        let result = state.entries.get_mut(&key).and_then(|e| {
            previous = Some(e.stamp);
            e.stamp = stamp;
            let CachedData::Page(points) = &e.data else {
                return None;
            };
            Some(points.clone())
        });
        if let Some(old) = previous {
            state.order.remove(&(old, key));
            state.order.insert((stamp, key));
        }
        if result.is_some() {
            state.stats.page_hits += 1;
        } else {
            state.stats.misses += 1;
        }
        result
    }
    fn insert(&self, key: CacheKey, bytes: u64, make: impl FnOnce() -> CachedData) {
        if self.config.max_bytes == 0 || bytes > self.config.max_bytes {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.entries.contains_key(&key) {
            return;
        }
        while bytes
            > self
                .config
                .max_bytes
                .saturating_sub(state.stats.resident_bytes)
        {
            if state.evict_one() == 0 {
                break;
            }
        }
        // Never call the budget's reclaim callback while holding this lock.
        // Optional cache insertion declines if other compute owns the slack.
        let lease = loop {
            if let Some(l) = self.memory.try_acquire_opportunistic(bytes) {
                break Some(l);
            }
            if state.evict_one() == 0 {
                break None;
            }
        };
        let Some(lease) = lease else {
            state.stats.admission_declined += 1;
            return;
        };
        state.stamp += 1;
        let stamp = state.stamp;
        state.entries.insert(
            key,
            CachedEntry {
                data: make(),
                stamp,
                bytes,
                _lease: lease,
            },
        );
        state.order.insert((stamp, key));
        state.stats.resident_bytes += bytes;
        state.stats.peak_resident_bytes = state
            .stats
            .peak_resident_bytes
            .max(state.stats.resident_bytes);
        state.stats.insertions += 1;
    }
    fn stats(&self) -> ReadCacheStats {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .stats
            .clone()
    }
}
fn decode_key(raw: &[u8]) -> Result<[usize; 3]> {
    let mut key = [0; 3];
    for i in 0..3 {
        key[i] = usize::try_from(u64::from_le_bytes(
            raw[i * 8..i * 8 + 8].try_into().unwrap(),
        ))
        .map_err(Error::backend)?;
    }
    Ok(key)
}

pub struct Reader {
    root: PathBuf,
    meta: Metadata,
    directory: File,
    cursor: u64,
    last: Option<[usize; 3]>,
    memory: MemoryBudget,
    cache: std::sync::Arc<ReadCache>,
}
impl Reader {
    pub fn open(
        root: impl Into<PathBuf>,
        expected_binding: &str,
        memory: MemoryBudget,
    ) -> Result<Self> {
        Self::open_cached(root, expected_binding, memory, ReadCacheConfig::default())
    }
    pub fn open_cached(
        root: impl Into<PathBuf>,
        expected_binding: &str,
        memory: MemoryBudget,
        cache: ReadCacheConfig,
    ) -> Result<Self> {
        if !(1..=1024).contains(&cache.directory_block_keys) {
            return Err(Error::invalid("invalid feature directory cache block size"));
        }
        let root = root.into();
        let path = root.join("manifest.json");
        if fs::metadata(&path).map_err(Error::backend)?.len() > META_LIMIT {
            return Err(Error::invalid("feature index manifest too large"));
        }
        let meta: Metadata = serde_json::from_slice(&fs::read(path).map_err(Error::backend)?)
            .map_err(Error::backend)?;
        check_meta(&meta)?;
        if !meta.sealed || meta.binding_sha256 != expected_binding {
            return Err(Error::invalid(
                "unsealed feature index or source binding mismatch",
            ));
        }
        let directory = File::open(root.join("cells.bin")).map_err(Error::backend)?;
        let expected = meta
            .occupied_cells
            .checked_mul(24)
            .ok_or_else(|| Error::invalid("feature directory overflow"))?;
        if directory.metadata().map_err(Error::backend)?.len() != expected {
            return Err(Error::invalid("truncated feature cell directory"));
        }
        if meta.version == 2 {
            let offsets = fs::metadata(root.join("page_offsets.bin")).map_err(Error::backend)?;
            let bytes = meta
                .occupied_cells
                .checked_mul(16)
                .ok_or_else(|| Error::invalid("packed offset directory overflow"))?;
            if offsets.len() != bytes {
                return Err(Error::invalid("truncated packed feature offset directory"));
            }
            let payload = fs::metadata(root.join("pages.bin")).map_err(Error::backend)?;
            if (meta.occupied_cells == 0) != (payload.len() == 0) {
                return Err(Error::invalid(
                    "packed feature payload disagrees with occupied directory",
                ));
            }
        }
        let cache = ReadCache::new(cache, memory.clone());
        Ok(Self {
            root,
            meta,
            directory,
            cursor: 0,
            last: None,
            memory,
            cache,
        })
    }
    pub fn cache_stats(&self) -> ReadCacheStats {
        self.cache.stats()
    }
    pub fn clear_cache(&self) {
        crate::budget::Reclaim::reclaim(&*self.cache, u64::MAX);
    }
    fn workspace(&self, bytes: u64, b: &mut Budget) -> Option<crate::budget::Lease> {
        if b.stop.is_some() {
            return None;
        }
        let hold = self.memory.try_acquire_reclaiming(bytes);
        if hold.is_none() {
            b.stop = Some(Stop::Memory);
        }
        b.stats.peak_planned_working_bytes = b
            .stats
            .peak_planned_working_bytes
            .max(self.memory.high_water());
        hold
    }
    fn directory_key(&self, index: u64, b: &mut Budget) -> Result<Outcome<[usize; 3]>> {
        let width = self.cache.config.directory_block_keys as u64;
        let block = index / width * width;
        if let Some(key) = self.cache.directory(block, (index - block) as usize) {
            return Ok(Outcome::Complete(key));
        }
        let count = width.min(self.meta.occupied_cells - block);
        let bytes = count * 24;
        // Optional read amplification is refused when it would consume a byte
        // budget the original single-key read could satisfy. Fall back to 24 B.
        let can_fill = self.cache.config.max_bytes >= bytes + 1024
            && bytes
                <= b.limits
                    .max_bytes_read
                    .saturating_sub(b.stats.charged_bytes_read);
        let hold = if can_fill {
            self.memory.try_acquire_reclaiming(bytes + 1024)
        } else {
            None
        };
        let fill = hold.is_some();
        let (start, len) = if fill { (block, bytes) } else { (index, 24) };
        if !b.directory_read(len) {
            return Ok(b.stopped());
        }
        let mut file = File::open(self.root.join("cells.bin")).map_err(Error::backend)?;
        file.seek(SeekFrom::Start(
            start
                .checked_mul(24)
                .ok_or_else(|| Error::invalid("directory offset overflow"))?,
        ))
        .map_err(Error::backend)?;
        let mut raw = vec![0u8; len as usize];
        file.read_exact(&mut raw).map_err(Error::backend)?;
        let at = ((index - start) * 24) as usize;
        let key = decode_key(&raw[at..at + 24])?;
        if fill {
            self.cache
                .insert(CacheKey::Directory(block), bytes + 1024, || {
                    CachedData::Directory(raw)
                });
            b.stats.peak_planned_working_bytes = b
                .stats
                .peak_planned_working_bytes
                .max(self.memory.high_water());
        }
        Ok(Outcome::Complete(key))
    }
    pub fn metadata(&self) -> &Metadata {
        &self.meta
    }
    pub fn next_cell(&mut self, b: &mut Budget) -> Result<Outcome<Option<[usize; 3]>>> {
        if b.stop.is_some() {
            return Ok(b.stopped());
        }
        if self.cursor == self.meta.occupied_cells {
            return Ok(Outcome::Complete(None));
        }
        let key = if self.cache.config.max_bytes > 0 {
            if !b.work(1) {
                return Ok(b.stopped());
            }
            match self.directory_key(self.cursor, b)? {
                Outcome::Complete(k) => k,
                Outcome::Stopped(_) => return Ok(b.stopped()),
            }
        } else {
            if !b.directory_read(24) || !b.work(1) {
                return Ok(b.stopped());
            }
            let mut raw = [0u8; 24];
            self.directory
                .read_exact(&mut raw)
                .map_err(Error::backend)?;
            let mut key = [0; 3];
            for i in 0..3 {
                key[i] = usize::try_from(u64::from_le_bytes(
                    raw[i * 8..i * 8 + 8].try_into().unwrap(),
                ))
                .map_err(Error::backend)?;
            }
            key
        };
        self.meta.geometry.cell_region(key)?;
        if self.last.is_some_and(|p| key <= p) {
            return Err(Error::invalid("feature cell directory is not canonical"));
        }
        self.cursor += 1;
        self.last = Some(key);
        Ok(Outcome::Complete(Some(key)))
    }
    fn contains_cell(&self, key: [usize; 3], b: &mut Budget) -> Result<Outcome<Option<u64>>> {
        if self.cache.config.max_bytes == 0 {
            return self.contains_cell_uncached(key, b);
        }
        let (mut low, mut high) = (0u64, self.meta.occupied_cells);
        while low < high {
            if !b.work(1) {
                return Ok(b.stopped());
            }
            b.stats.directory_probes += 1;
            let mid = low + (high - low) / 2;
            let found = match self.directory_key(mid, b)? {
                Outcome::Complete(k) => k,
                Outcome::Stopped(_) => return Ok(b.stopped()),
            };
            match found.cmp(&key) {
                Ordering::Equal => return Ok(Outcome::Complete(Some(mid))),
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
            }
        }
        Ok(Outcome::Complete(None))
    }
    fn contains_cell_uncached(
        &self,
        key: [usize; 3],
        b: &mut Budget,
    ) -> Result<Outcome<Option<u64>>> {
        // A separate file description avoids changing the streaming cursor or
        // sharing seek offsets between independent read-only queries.
        let mut file = File::open(self.root.join("cells.bin")).map_err(Error::backend)?;
        let (mut low, mut high) = (0u64, self.meta.occupied_cells);
        while low < high {
            if !b.directory_read(24) || !b.work(1) {
                return Ok(b.stopped());
            }
            b.stats.directory_probes += 1;
            let mid = low + (high - low) / 2;
            file.seek(SeekFrom::Start(
                mid.checked_mul(24)
                    .ok_or_else(|| Error::invalid("directory offset overflow"))?,
            ))
            .map_err(Error::backend)?;
            let mut raw = [0; 24];
            file.read_exact(&mut raw).map_err(Error::backend)?;
            let mut found = [0usize; 3];
            for i in 0..3 {
                found[i] = usize::try_from(u64::from_le_bytes(
                    raw[i * 8..i * 8 + 8].try_into().unwrap(),
                ))
                .map_err(Error::backend)?;
            }
            match found.cmp(&key) {
                Ordering::Equal => return Ok(Outcome::Complete(Some(mid))),
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
            }
        }
        Ok(Outcome::Complete(None))
    }
    pub fn read_cell(
        &self,
        key: [usize; 3],
        b: &mut Budget,
    ) -> Result<Outcome<Vec<LocatedFeature>>> {
        self.read_cell_extra(key, 0, b)
    }
    fn read_cell_extra(
        &self,
        key: [usize; 3],
        extra: usize,
        b: &mut Budget,
    ) -> Result<Outcome<Vec<LocatedFeature>>> {
        let region = self.meta.geometry.cell_region(key)?;
        if !b.cell() {
            return Ok(b.stopped());
        }
        // A failed optional page-copy reservation must not turn an empty
        // directory cell into a memory error. Confirm membership before
        // requiring page workspace when a cache hit cannot be admitted.
        let _cached_hold = if self.cache.config.max_bytes > 0 {
            self.memory
                .try_acquire_reclaiming(plan(self.meta.per_cell_limit, extra)?)
        } else {
            None
        };
        b.stats.peak_planned_working_bytes = b
            .stats
            .peak_planned_working_bytes
            .max(self.memory.high_water());
        if _cached_hold.is_some() {
            if let Some(points) = self.cache.page(key) {
                if !b.work(points.len() as u64) {
                    return Ok(b.stopped());
                }
                return Ok(Outcome::Complete(points));
            }
        }
        let ordinal = match self.contains_cell(key, b)? {
            Outcome::Stopped(_) => return Ok(b.stopped()),
            Outcome::Complete(None) => return Ok(Outcome::Complete(vec![])),
            Outcome::Complete(Some(index)) => index,
        };
        let path = page(&self.root, key);
        let (bytes, mut packed_file) = if self.meta.version == 2 {
            if !b.directory_read(16) {
                return Ok(b.stopped());
            }
            let mut offsets =
                File::open(self.root.join("page_offsets.bin")).map_err(Error::backend)?;
            offsets
                .seek(SeekFrom::Start(ordinal.checked_mul(16).ok_or_else(
                    || Error::invalid("packed offset position overflow"),
                )?))
                .map_err(Error::backend)?;
            let mut raw = [0u8; 16];
            offsets.read_exact(&mut raw).map_err(Error::backend)?;
            let offset = u64::from_le_bytes(raw[..8].try_into().unwrap());
            let bytes = u64::from_le_bytes(raw[8..].try_into().unwrap());
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid("packed feature range overflow"))?;
            let mut file = File::open(self.root.join("pages.bin")).map_err(Error::backend)?;
            if end > file.metadata().map_err(Error::backend)?.len() {
                return Err(Error::invalid("packed feature range exceeds payload"));
            }
            file.seek(SeekFrom::Start(offset)).map_err(Error::backend)?;
            (bytes, Some(file))
        } else {
            let info = match fs::metadata(&path) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(Error::invalid(
                        "sealed feature index is missing an indexed page",
                    ))
                }
                Err(e) => return Err(Error::backend(e)),
            };
            (info.len(), None)
        };
        let header = encoded_header_bytes(&schema());
        let width = encoded_row_bytes(&schema());
        if bytes < header
            || (bytes - header) % width != 0
            || (bytes - header) / width > self.meta.per_cell_limit as u64
        {
            return Err(Error::invalid(
                "feature page size violates schema/cell bound",
            ));
        }
        let rows = (bytes - header) / width;
        if !b.work(rows * (rows.max(1).ilog2() as u64 + 3)) {
            return Ok(b.stopped());
        }
        let _uncached_hold = if _cached_hold.is_none() {
            let Some(hold) = self.workspace(plan(self.meta.per_cell_limit, extra)?, b) else {
                return Ok(b.stopped());
            };
            Some(hold)
        } else {
            None
        };
        if !b.page_read(bytes) {
            return Ok(b.stopped());
        }
        let data = if let Some(file) = &mut packed_file {
            let len = usize::try_from(bytes).map_err(Error::backend)?;
            let mut data = vec![0u8; len];
            file.read_exact(&mut data).map_err(Error::backend)?;
            data
        } else {
            fs::read(path).map_err(Error::backend)?
        };
        if data.len() as u64 != bytes {
            return Err(Error::invalid("feature page changed during read"));
        }
        b.stats.page_decodes += 1;
        let mut table = Table::new(self.meta.geometry.shape_zyx, schema())?;
        table.write(key, &data)?;
        if table.len() as u64 != rows {
            return Err(Error::invalid("feature page row count mismatch"));
        }
        table.seal_as(Layout::Flat)?;
        let mut out = Vec::with_capacity(rows as usize);
        for row in table.scan(&region)? {
            let feature = Feature {
                at_zyx: row.at(),
                position_xyz: [row.f64(0)?, row.f64(1)?, row.f64(2)?],
                weight: row.f64(3)?,
                scale: row.f64(4)?,
            }
            .checked(&self.meta.geometry, key)?;
            out.push(LocatedFeature {
                key: FeatureKey {
                    cell_zyx: key,
                    row: out.len(),
                },
                feature,
            });
        }
        if out.len() != rows as usize {
            return Err(Error::invalid(
                "feature page contains rows outside its cell",
            ));
        }
        self.cache
            .insert(CacheKey::Page(key), 1024 + out.len() as u64 * 128, || {
                CachedData::Page(out.clone())
            });
        b.stats.peak_planned_working_bytes = b
            .stats
            .peak_planned_working_bytes
            .max(self.memory.high_water());
        Ok(Outcome::Complete(out))
    }
    /// Exact radius filter followed by deterministic nearest-K retention. A
    /// successful truncated result is a policy cap, not an empty neighborhood.
    /// A resource stop discards the query's partial points. Returned buffers are
    /// caller-owned after the operation's shared memory lease is released.
    pub fn neighbors(
        &self,
        position: [f64; 3],
        radius: f64,
        limit: usize,
        b: &mut Budget,
    ) -> Result<Outcome<Neighbors>> {
        self.neighbors_by(
            position,
            radius,
            1,
            limit,
            |_, distance| Some((0, distance)),
            b,
        )
    }
    /// Partition and rank points BEFORE bounded retention. The selector must be
    /// deterministic and bounded-cost; smaller finite priority wins, with
    /// FeatureKey ties. None excludes a point. A bin cap is not redistributed
    /// to other bins. At most 256 partitions and 4096 total retained points.
    /// Work/IO limits include all visited points, including excluded ones.
    pub fn neighbors_by(
        &self,
        position: [f64; 3],
        radius: f64,
        partitions: usize,
        per_partition: usize,
        mut select: impl FnMut(&LocatedFeature, f64) -> Option<(usize, f64)>,
        b: &mut Budget,
    ) -> Result<Outcome<Neighbors>> {
        let limit = partitions
            .checked_mul(per_partition)
            .ok_or_else(|| Error::invalid("feature neighborhood capacity overflow"))?;
        if partitions == 0 || partitions > 256 || per_partition == 0 || limit > 4096 {
            return Err(Error::invalid(
                "invalid partitioned feature neighborhood capacity",
            ));
        }
        if !radius.is_finite()
            || radius <= 0.
            || radius * radius == f64::INFINITY
            || position.iter().any(|v| !v.is_finite())
            || limit == 0
            || limit > 4096
        {
            return Err(Error::invalid("invalid feature neighborhood"));
        }
        if b.stop.is_some() {
            return Ok(b.stopped());
        }
        let g = &self.meta.geometry;
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for i in 0..3 {
            let q = (position[i] - g.origin_xyz[i]) / g.spacing_xyz[i];
            let mut pad = radius / g.spacing_xyz[i] + g.max_offset_voxels;
            if !q.is_finite() || !pad.is_finite() {
                return Err(Error::invalid("feature neighborhood coordinate overflow"));
            }
            // Include query conversion and admitted point-construction roundoff
            // in broad-phase bounds, including tiny queries at cell seams.
            // The final physical-distance filter is unchanged.
            pad += 2.
                * anchor_roundoff_voxels(
                    g.origin_xyz[i],
                    position[i],
                    g.spacing_xyz[i],
                    q.abs() + pad + 1.,
                )?;
            if q + pad < 0. || q - pad > (g.shape_zyx[2 - i] - 1) as f64 {
                return Ok(Outcome::Complete(Neighbors {
                    points: vec![],
                    truncated: false,
                    matching_points: 0,
                    radius_points: 0,
                    partition_counts: vec![0; partitions],
                }));
            }
            let anchor_low = (q - pad).ceil().max(0.);
            let anchor_high = (q + pad).floor().min((g.shape_zyx[2 - i] - 1) as f64);
            if anchor_low > anchor_high {
                return Ok(Outcome::Complete(Neighbors {
                    points: vec![],
                    truncated: false,
                    matching_points: 0,
                    radius_points: 0,
                    partition_counts: vec![0; partitions],
                }));
            }
            lo[2 - i] = (anchor_low as usize / g.cell_zyx[2 - i]).min(g.cells_zyx()[2 - i] - 1);
            hi[2 - i] = (anchor_high as usize / g.cell_zyx[2 - i]).min(g.cells_zyx()[2 - i] - 1);
        }
        let cells = (0..3)
            .try_fold(1u64, |a, i| a.checked_mul((hi[i] - lo[i] + 1) as u64))
            .ok_or_else(|| Error::invalid("feature query cell count overflow"))?;
        if cells > b.limits.max_cell_visits.saturating_sub(b.stats.cell_visits) {
            b.stop = Some(Stop::Cells);
            return Ok(b.stopped());
        }
        let Some(_hold) = self.workspace(plan(0, limit)?, b) else {
            return Ok(b.stopped());
        };
        let mut nearest: Vec<(usize, f64, LocatedFeature)> = Vec::with_capacity(limit);
        let mut matching = 0u64;
        let mut radius_points = 0u64;
        let mut counts = vec![0u64; partitions];
        for z in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for x in lo[2]..=hi[2] {
                    let points = match self.read_cell_extra([z, y, x], 0, b)? {
                        Outcome::Complete(p) => p,
                        Outcome::Stopped(_) => return Ok(b.stopped()),
                    };
                    for point in points {
                        if !b.work(1 + 2 * nearest.len() as u64) {
                            return Ok(b.stopped());
                        }
                        let distance = point
                            .feature
                            .position_xyz
                            .iter()
                            .zip(position)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();
                        if distance <= radius * radius {
                            radius_points = radius_points
                                .checked_add(1)
                                .ok_or_else(|| Error::invalid("feature radius count overflow"))?;
                            let Some((partition, priority)) = select(&point, distance) else {
                                continue;
                            };
                            if partition >= partitions || !priority.is_finite() {
                                return Err(Error::invalid(
                                    "invalid feature selection partition/priority",
                                ));
                            }
                            matching = matching
                                .checked_add(1)
                                .ok_or_else(|| Error::invalid("feature match count overflow"))?;
                            counts[partition] =
                                counts[partition].checked_add(1).ok_or_else(|| {
                                    Error::invalid("feature partition count overflow")
                                })?;
                            let candidate = (partition, priority, point);
                            let compare =
                                |a: &(usize, f64, LocatedFeature),
                                 z: &(usize, f64, LocatedFeature)| {
                                    a.0.cmp(&z.0)
                                        .then(a.1.total_cmp(&z.1))
                                        .then(a.2.key.cmp(&z.2.key))
                                };
                            if counts[partition] <= per_partition as u64 {
                                nearest.push(candidate);
                            } else {
                                let worst = (0..nearest.len())
                                    .filter(|&i| nearest[i].0 == partition)
                                    .max_by(|&a, &z| compare(&nearest[a], &nearest[z]))
                                    .expect("nonempty selected partition");
                                if compare(&candidate, &nearest[worst]).is_lt() {
                                    nearest[worst] = candidate;
                                }
                            }
                        }
                    }
                }
            }
        }
        nearest.sort_unstable_by(|a, z| {
            a.0.cmp(&z.0)
                .then(a.1.total_cmp(&z.1))
                .then(a.2.key.cmp(&z.2.key))
        });
        Ok(Outcome::Complete(Neighbors {
            points: nearest.into_iter().map(|(_, _, p)| p).collect(),
            truncated: counts.iter().any(|&n| n > per_partition as u64),
            matching_points: matching,
            radius_points,
            partition_counts: counts,
        }))
    }
}
