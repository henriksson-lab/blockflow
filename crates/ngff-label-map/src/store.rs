//! The directory store: `snapshot.bin` + append-only `edits.jsonl`, a writer lock, and
//! discovery through the label group's metadata. See the crate docs for the layout.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value as Json};

use crate::history::{EditRecord, SegmentMapState};
use crate::map::SegmentMap;
use crate::snapshot::{self, HEADER_LEN};
use crate::{Error, Result};

pub const SNAPSHOT_FILE: &str = "snapshot.bin";
pub const EDITS_FILE: &str = "edits.jsonl";
/// The writer lock is a sibling file, `.<store directory name>.lock`, so it exists before the
/// store does and creation can be serialised too.
pub const LOCK_SUFFIX: &str = ".lock";
/// The key under the label group's `attributes` (Zarr v3 `zarr.json`, or v2 `.zattrs`).
pub const ATTRIBUTE: &str = "segment_map";
/// The conventional store directory inside a label group.
pub const DEFAULT_DIR: &str = "segment-map";
pub const ATTRIBUTE_VERSION: u32 = 1;
/// A snapshot is rewritten after this many edits beyond the current one, bounding replay.
pub const DEFAULT_SNAPSHOT_INTERVAL: u64 = 64;

/// Who and when, recorded on each edit.
#[derive(Clone, Debug, Default)]
pub struct EditMeta {
    pub author: Option<String>,
    /// Defaults to now.
    pub time_ms: Option<u64>,
}

/// A store directory.
#[derive(Clone, Debug)]
pub struct SegmentMapStore {
    dir: PathBuf,
    snapshot_interval: u64,
}

/// A loaded map plus the position in the store it was read up to, so it can be refreshed by
/// reading only the appended log lines.
#[derive(Clone, Debug)]
pub struct LoadedSegmentMap {
    state: SegmentMapState,
    generation: u64,
    snapshot_revision: u64,
    edits_offset: u64,
}

impl LoadedSegmentMap {
    pub fn state(&self) -> &SegmentMapState {
        &self.state
    }

    pub fn map(&self) -> &SegmentMap {
        self.state.map()
    }

    pub fn revision(&self) -> u64 {
        self.state.revision()
    }

    /// Identity of the map's lineage: a fresh [`SegmentMapStore::create`] (for example a
    /// re-run pipeline) changes it, compaction does not.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn snapshot_revision(&self) -> u64 {
        self.snapshot_revision
    }
}

/// Holds the store's writer lock until dropped.
pub struct WriterLock {
    _file: File,
}

impl SegmentMapStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            snapshot_interval: DEFAULT_SNAPSHOT_INTERVAL,
        }
    }

    pub fn with_snapshot_interval(mut self, edits: u64) -> Self {
        self.snapshot_interval = edits.max(1);
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn exists(&self) -> bool {
        self.dir.join(SNAPSHOT_FILE).is_file()
    }

    /// Create a store at `dir` holding `map` at revision 0. The directory appears complete or
    /// not at all (written beside it, then renamed). An existing store is refused.
    pub fn create(dir: impl Into<PathBuf>, map: &SegmentMap) -> Result<Self> {
        let store = Self::new(dir);
        if !store.create_if_missing(|| Ok(map.clone()))? {
            return Err(Error::Invalid(format!(
                "a segment map already exists at {}",
                store.dir.display()
            )));
        }
        Ok(store)
    }

    /// Create the store from `init()` unless it exists; returns whether it was created.
    /// Serialised with writers by the lock, so concurrent first edits create it once.
    pub fn create_if_missing(&self, init: impl FnOnce() -> Result<SegmentMap>) -> Result<bool> {
        let _lock = self.lock()?;
        if self.exists() {
            return Ok(false);
        }
        let bytes = snapshot::encode(&init()?, 0, new_generation(&self.dir));
        self.install(&[(SNAPSHOT_FILE, &bytes), (EDITS_FILE, b"")])?;
        Ok(true)
    }

    /// Copy `source`'s current state and full history here unless a store exists (for example
    /// from a read-only dataset into a writable state directory on the first edit); returns
    /// whether it copied. The copy is a new lineage.
    pub fn copy_if_missing(&self, source: &SegmentMapStore) -> Result<bool> {
        let _lock = self.lock()?;
        if self.exists() {
            return Ok(false);
        }
        let loaded = source.load()?;
        let bytes = snapshot::encode(loaded.map(), loaded.revision(), new_generation(&self.dir));
        let edits = read_complete_lines(&source.dir.join(EDITS_FILE))?.0;
        self.install(&[(SNAPSHOT_FILE, &bytes), (EDITS_FILE, &edits)])?;
        Ok(true)
    }

    /// Write `files` into the store directory atomically. Call with the lock held.
    fn install(&self, files: &[(&str, &[u8])]) -> Result<()> {
        let (parent, name) = self.parent_and_name()?;
        fs::create_dir_all(parent)?;
        let staging = parent.join(format!(
            ".{name}.staging-{:016x}",
            new_generation(&self.dir)
        ));
        fs::create_dir_all(&staging)?;
        let result = (|| {
            for (file, bytes) in files {
                let mut out = File::create(staging.join(file))?;
                out.write_all(bytes)?;
                out.sync_all()?;
            }
            // `rename` replaces an empty directory but not a populated one.
            if self.dir.is_dir() && fs::read_dir(&self.dir)?.next().is_some() {
                return Err(Error::Invalid(format!(
                    "{} exists and is not an empty directory",
                    self.dir.display()
                )));
            }
            fs::rename(&staging, &self.dir)?;
            sync_dir(parent);
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    /// Read the map: the snapshot, then every complete log line after it. Needs no lock: the
    /// snapshot is replaced atomically and the log only grows, so any snapshot read is
    /// consistent with the log read after it.
    pub fn load(&self) -> Result<LoadedSegmentMap> {
        let bytes = fs::read(self.dir.join(SNAPSHOT_FILE)).map_err(|error| {
            Error::Io(std::io::Error::new(
                error.kind(),
                format!("read {}: {error}", self.dir.join(SNAPSHOT_FILE).display()),
            ))
        })?;
        let (header, map) = snapshot::decode(&bytes)?;
        let (edits, offset) = read_complete_lines(&self.dir.join(EDITS_FILE))?;
        let records = parse_records(&edits)?;
        let state = SegmentMapState::from_snapshot(map, header.revision, records)?;
        Ok(LoadedSegmentMap {
            state,
            generation: header.generation,
            snapshot_revision: header.revision,
            edits_offset: offset,
        })
    }

    /// Bring `loaded` up to date; returns whether anything changed. Costs one header read and
    /// one `stat` when nothing did, and reads only the appended lines when edits were added.
    pub fn refresh(&self, loaded: &mut LoadedSegmentMap) -> Result<bool> {
        let mut header = [0u8; HEADER_LEN];
        File::open(self.dir.join(SNAPSHOT_FILE))?.read_exact(&mut header)?;
        let header = snapshot::decode_header(&header)?;
        let edits_path = self.dir.join(EDITS_FILE);
        let len = fs::metadata(&edits_path)?.len();
        if header.generation != loaded.generation || len < loaded.edits_offset {
            *loaded = self.load()?;
            return Ok(true);
        }
        if len == loaded.edits_offset {
            return Ok(false);
        }
        let mut file = File::open(&edits_path)?;
        file.seek(SeekFrom::Start(loaded.edits_offset))?;
        let mut tail = Vec::new();
        file.read_to_end(&mut tail)?;
        let complete = tail
            .iter()
            .rposition(|b| *b == b'\n')
            .map_or(0, |at| at + 1);
        if complete == 0 {
            return Ok(false);
        }
        let mut next = loaded.state.clone();
        for record in parse_records(&tail[..complete])? {
            next.commit(record)?;
        }
        loaded.state = next;
        loaded.edits_offset += complete as u64;
        Ok(true)
    }

    /// Take the writer lock (blocking). Every edit and creation takes it; readers never do.
    pub fn lock(&self) -> Result<WriterLock> {
        let (parent, name) = self.parent_and_name()?;
        fs::create_dir_all(parent)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(parent.join(format!(".{name}{LOCK_SUFFIX}")))?;
        file.lock()?;
        Ok(WriterLock { _file: file })
    }

    fn parent_and_name(&self) -> Result<(&Path, &str)> {
        let parent = self
            .dir
            .parent()
            .ok_or_else(|| Error::Invalid("segment map directory has no parent".into()))?;
        let name = self
            .dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| Error::Invalid("segment map directory has no name".into()))?;
        Ok((parent, name))
    }

    /// Merge segments `a` and `b` if the store is at `expected_revision`.
    pub fn merge(
        &self,
        loaded: &mut LoadedSegmentMap,
        expected_revision: u64,
        a: u64,
        b: u64,
        meta: EditMeta,
    ) -> Result<EditRecord> {
        self.edit(loaded, expected_revision, meta, |state| {
            state.plan_merge(a, b)
        })
    }

    /// Put `fragments` into `segment` if the store is at `expected_revision`.
    pub fn assign(
        &self,
        loaded: &mut LoadedSegmentMap,
        expected_revision: u64,
        fragments: &[u64],
        segment: u64,
        meta: EditMeta,
    ) -> Result<EditRecord> {
        self.edit(loaded, expected_revision, meta, |state| {
            state.plan_assign(fragments, segment)
        })
    }

    /// Undo the most recent live edit if the store is at `expected_revision`.
    pub fn undo(
        &self,
        loaded: &mut LoadedSegmentMap,
        expected_revision: u64,
        meta: EditMeta,
    ) -> Result<EditRecord> {
        self.edit(loaded, expected_revision, meta, SegmentMapState::plan_undo)
    }

    /// The general edit transaction: under the writer lock, refresh `loaded` from disk, refuse
    /// with [`Error::Conflict`] unless it is at `expected_revision`, plan the record, append it
    /// durably, then apply it in memory (and compact when due). `loaded` is left current.
    pub fn edit(
        &self,
        loaded: &mut LoadedSegmentMap,
        expected_revision: u64,
        meta: EditMeta,
        plan: impl FnOnce(&SegmentMapState) -> Result<EditRecord>,
    ) -> Result<EditRecord> {
        let _lock = self.lock()?;
        self.refresh(loaded)?;
        if loaded.revision() != expected_revision {
            return Err(Error::Conflict {
                expected: expected_revision,
                current: loaded.revision(),
            });
        }
        let mut record = plan(&loaded.state)?;
        record.author = meta.author;
        record.time_ms = meta.time_ms.or_else(now_ms);
        let mut next = loaded.state.clone();
        next.commit(record.clone())?;
        let mut line = serde_json::to_vec(&record).map_err(|e| Error::Corrupt(e.to_string()))?;
        line.push(b'\n');
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.dir.join(EDITS_FILE))?;
        // Drop a torn line a crashed writer may have left; it was never a committed edit.
        file.set_len(loaded.edits_offset)?;
        file.seek(SeekFrom::Start(loaded.edits_offset))?;
        file.write_all(&line)?;
        file.sync_data()?;
        loaded.state = next;
        loaded.edits_offset += line.len() as u64;
        if loaded.revision() - loaded.snapshot_revision >= self.snapshot_interval {
            self.write_snapshot(loaded)?;
        }
        Ok(record)
    }

    /// Rewrite the snapshot at `loaded`'s revision (atomic replace). Call with the writer lock
    /// held, or through [`Self::compact`].
    fn write_snapshot(&self, loaded: &mut LoadedSegmentMap) -> Result<()> {
        let bytes = snapshot::encode(loaded.map(), loaded.revision(), loaded.generation);
        write_atomic(&self.dir.join(SNAPSHOT_FILE), &bytes)?;
        loaded.snapshot_revision = loaded.revision();
        Ok(())
    }

    /// Fold every edit into the snapshot now.
    pub fn compact(&self, loaded: &mut LoadedSegmentMap) -> Result<()> {
        let _lock = self.lock()?;
        self.refresh(loaded)?;
        self.write_snapshot(loaded)
    }
}

/// The store a label group declares in its metadata, if any:
/// `attributes.segment_map = {"version": 1, "path": "segment-map"}` in `zarr.json` (or the same
/// key in a Zarr v2 `.zattrs`). The path is relative to the label group and may not leave it.
pub fn discover(label_dir: &Path) -> Result<Option<SegmentMapStore>> {
    let attribute = [label_dir.join("zarr.json"), label_dir.join(".zattrs")]
        .iter()
        .filter(|path| path.is_file())
        .find_map(|path| {
            let value: Json = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
            let attributes = if path.ends_with("zarr.json") {
                value.get("attributes")?
            } else {
                &value
            };
            attributes.get(ATTRIBUTE).cloned()
        });
    let Some(attribute) = attribute else {
        return Ok(None);
    };
    let version = attribute.get("version").and_then(Json::as_u64).unwrap_or(0);
    if version != u64::from(ATTRIBUTE_VERSION) {
        return Err(Error::Invalid(format!(
            "segment_map version {version} is not supported (expected {ATTRIBUTE_VERSION})"
        )));
    }
    let relative = attribute
        .get("path")
        .and_then(Json::as_str)
        .unwrap_or(DEFAULT_DIR);
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || !relative_path
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(Error::Invalid(format!(
            "segment_map path {relative:?} must be relative and stay inside the label group"
        )));
    }
    Ok(Some(SegmentMapStore::new(label_dir.join(relative_path))))
}

/// Declare `relative` as the label group's segment map in its `zarr.json` (atomic rewrite that
/// keeps every other key).
pub fn write_attribute(label_dir: &Path, relative: &str) -> Result<()> {
    let path = label_dir.join("zarr.json");
    let mut value: Json = serde_json::from_slice(&fs::read(&path)?)
        .map_err(|e| Error::Corrupt(format!("{}: {e}", path.display())))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| Error::Corrupt(format!("{} is not an object", path.display())))?;
    let attributes = object.entry("attributes").or_insert_with(|| json!({}));
    let attributes = attributes
        .as_object_mut()
        .ok_or_else(|| Error::Corrupt("zarr.json attributes is not an object".into()))?;
    attributes.insert(
        ATTRIBUTE.into(),
        json!({"version": ATTRIBUTE_VERSION, "path": relative}),
    );
    let bytes = serde_json::to_vec_pretty(&value).map_err(|e| Error::Corrupt(e.to_string()))?;
    write_atomic(&path, &bytes)
}

/// Create `label_dir/segment-map` holding `map` and declare it in the label's metadata. This is
/// what a pipeline calls after writing fragments and the initial (multicut) grouping.
pub fn create_for_label(label_dir: &Path, map: &SegmentMap) -> Result<SegmentMapStore> {
    let store = SegmentMapStore::create(label_dir.join(DEFAULT_DIR), map)?;
    write_attribute(label_dir, DEFAULT_DIR)?;
    Ok(store)
}

fn parse_records(bytes: &[u8]) -> Result<Vec<EditRecord>> {
    bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .map(|line| {
            serde_json::from_slice(line).map_err(|e| Error::Corrupt(format!("{EDITS_FILE}: {e}")))
        })
        .collect()
}

/// The file's bytes up to and including its last newline, and that length. A missing file is
/// empty.
fn read_complete_lines(path: &Path) -> Result<(Vec<u8>, u64)> {
    let mut bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let complete = bytes
        .iter()
        .rposition(|b| *b == b'\n')
        .map_or(0, |at| at + 1);
    bytes.truncate(complete);
    Ok((bytes, complete as u64))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("path has no parent".into()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Invalid("path has no file name".into()))?;
    let temporary = parent.join(format!(".{name}.tmp-{:016x}", new_generation(path)));
    let result = (|| {
        let mut out = File::create(&temporary)?;
        out.write_all(bytes)?;
        out.sync_all()?;
        fs::rename(&temporary, path)?;
        sync_dir(parent);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Best effort: make a rename durable. Not every platform can open a directory.
fn sync_dir(dir: &Path) {
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
}

fn now_ms() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// A practically unique id from the clock, the process and a path (no RNG dependency).
fn new_generation(salt: &Path) -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in salt.to_string_lossy().bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    splitmix(nanos ^ splitmix(u64::from(std::process::id()) ^ hash) ^ splitmix(count))
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
